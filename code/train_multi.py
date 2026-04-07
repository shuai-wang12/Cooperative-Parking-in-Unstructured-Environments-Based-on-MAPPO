from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from multi_env import MultiVehicleLiDAR2DEnv, SubprocMultiAgentVecEnv
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from gymnasium import spaces

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import random

def set_global_seeds(seed):
    """
    确保实验可复现：固定 random, numpy, torch, sb3 的种子
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(seed)

# --- MAPPO 核心架构设计 (CTDE) ---

class MAPPOFeatureExtractor(BaseFeaturesExtractor):
    """
    MAPPO 特征提取器：
    将 Dict 观测中的 'obs' (局部) 和 'state' (全局) 分别预处理。
    输出为拼接向量，待后续 Policy 中进行切片分发。
    """
    def __init__(self, observation_space: spaces.Dict):
        # 先计算总维度
        total_concat_size = 0
        for subspace in observation_space.spaces.values():
            total_concat_size += 256
            
        super().__init__(observation_space, features_dim=total_concat_size)
        
        extractors = {}
        for key, subspace in observation_space.spaces.items():
            extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 256), nn.ReLU())
            
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: dict) -> torch.Tensor:
        encoded_tensor_list = []
        # 确保顺序一致：先 obs 再 state
        for key in ["obs", "state"]:
            encoded_tensor_list.append(self.extractors[key](observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)

class MAPPOMlpExtractor(nn.Module):
    """
    自定义 MLP 提取器 (CTDE):
    - Actor 只看局部编码 (前 256 维)
    - Critic 看局部 + 全局编码 (全部 512 维)
    """
    def __init__(self, feature_dim: int, last_layer_dim_pi: int, last_layer_dim_vf: int):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        half_dim = feature_dim // 2  # 256
        
        # Policy 网络: 只处理前 256 维局部特征
        self.policy_net = nn.Sequential(
            nn.Linear(half_dim, 256), nn.Tanh(),
            nn.Linear(256, last_layer_dim_pi), nn.Tanh()
        )
        
        # Value 网络: 必须看到局部+全局 (Obs + State)
        # 即使 State 包含全局信息，Critic 的梯度也需要回传给 Obs 编码器
        # 帮助 Actor 训练其特征提取层 ("Shared Trunk" 效应)
        # 🚨 MAPPO 专属优化: 加宽 Critic 网络 (256 -> 512) 以处理复杂的全局交互信息
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.Tanh(),
            nn.Linear(512, last_layer_dim_vf), nn.Tanh()
        )

    def forward(self, features: torch.Tensor):
        half = features.shape[1] // 2
        # PyTorch 的切片操作会自然地将梯度只传递给 obs 编码器
        # 不需要 detach，切片本身就隔离了梯度流向
        local_feat = features[:, :half]
        # Critic 接收完整特征 (Shared Trunk)
        return self.policy_net(local_feat), self.value_net(features)
        
    def forward_actor(self, features: torch.Tensor):
        half = features.shape[1] // 2
        local_feat = features[:, :half]
        return self.policy_net(local_feat)

    def forward_critic(self, features: torch.Tensor):
        return self.value_net(features)

class MAPPOPolicy(MultiInputActorCriticPolicy):
    """
    自定义 MAPPO 策略类：集成中心化评论家。
    """
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MAPPOMlpExtractor(self.features_dim, 256, 256)

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

class EpisodeImageCallback(BaseCallback):
    """
    自定义回调：每隔一定步数（默认1000步）保存环境渲染图，并记录成功率和碰撞率。
    同时追踪最佳模型（成功率最高且碰撞率最低）并自动保存。
    """
    def __init__(self, save_dir, save_freq_steps=1000, verbose=0):
        super(EpisodeImageCallback, self).__init__(verbose)
        self.save_dir = save_dir
        self.save_freq_steps = save_freq_steps
        self.episode_count = 0 
        self.success_count = 0
        self.recent_successes = []
        self.recent_collisions = []
        self.recent_timeouts = []
        self.recent_rewards = []
        # 三种超时分类追踪
        self.recent_w_misses = []
        self.recent_rev_stucks = []
        self.recent_global_limits = []
        self.last_save_step = 0
        self.current_episode_rewards = None
        self.best_window_score = -float('inf')
        
        os.makedirs(save_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        # 熵系数线性衰减：从 0.01 线性降到 0.001
        progress = 1.0 - (self.num_timesteps / self.model._total_timesteps)
        self.model.ent_coef = 0.001 + 0.009 * progress  # 0.01 -> 0.001
        
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        rewards = self.locals.get("rewards")
        
        # 初始化累计奖励数组
        if self.current_episode_rewards is None and dones is not None:
             self.current_episode_rewards = np.zeros(len(dones), dtype=np.float32)
             
        if rewards is not None and self.current_episode_rewards is not None:
            self.current_episode_rewards += rewards

        if dones is not None:
            for i, done in enumerate(dones):
                if done:
                    is_success = infos[i].get('success', False)
                    is_collision = infos[i].get('collision', False)
                    is_timeout = infos[i].get('timeout', False)
                    
                    # 跳过空槽位的幽灵回合（inactive agent 被 dones.fill(True) 强制结束）
                    if not is_success and not is_collision and not is_timeout:
                        self.current_episode_rewards[i] = 0.0
                        continue
                    
                    self.episode_count += 1
                    
                    # 记录这一回合的总奖励
                    ep_reward = self.current_episode_rewards[i]
                    self.recent_rewards.append(ep_reward)
                    self.current_episode_rewards[i] = 0.0 # 重置
                    
                    if is_success:
                        self.success_count += 1
                    
                    self.recent_successes.append(1.0 if is_success else 0.0)
                    self.recent_collisions.append(1.0 if is_collision else 0.0)
                    self.recent_timeouts.append(1.0 if is_timeout else 0.0)
                    
                    # 三种超时精准分类
                    timeout_reason = infos[i].get('timeout_reason', '')
                    self.recent_w_misses.append(1.0 if timeout_reason == 'w_point_miss' else 0.0)
                    self.recent_rev_stucks.append(1.0 if timeout_reason == 'reversing_stuck' else 0.0)
                    self.recent_global_limits.append(1.0 if timeout_reason == 'global_limit' else 0.0)
                    
                    for lst in [self.recent_successes, self.recent_collisions, self.recent_timeouts,
                                self.recent_rewards, self.recent_w_misses, self.recent_rev_stucks, self.recent_global_limits]:
                        if len(lst) > 3000: lst.pop(0)

                    if len(self.recent_successes) > 0:
                        success_rate = sum(self.recent_successes) / len(self.recent_successes)
                        collision_rate = sum(self.recent_collisions) / len(self.recent_collisions)
                        timeout_rate = sum(self.recent_timeouts) / len(self.recent_timeouts)
                        reward_mean = sum(self.recent_rewards) / len(self.recent_rewards)
                        
                        self.logger.record("train/success_rate", success_rate)
                        self.logger.record("train/collision_rate", collision_rate)
                        self.logger.record("train/timeout_rate", timeout_rate)
                        self.logger.record("train/reward_mean", reward_mean)
                        
                        # 三种超时分类日志
                        w_miss_rate = sum(self.recent_w_misses) / len(self.recent_w_misses)
                        rev_stuck_rate = sum(self.recent_rev_stucks) / len(self.recent_rev_stucks)
                        global_limit_rate = sum(self.recent_global_limits) / len(self.recent_global_limits)
                        self.logger.record("timeout/w_point_miss", w_miss_rate)
                        self.logger.record("timeout/reversing_stuck", rev_stuck_rate)
                        self.logger.record("timeout/global_limit", global_limit_rate)
                        
                        # 滑动窗口最佳模型保存 (至少200样本)
                        if len(self.recent_successes) >= 200:
                            window_score = success_rate - 5.0 * collision_rate - 0.5 * timeout_rate
                            if window_score > self.best_window_score:
                                self.best_window_score = window_score
                                self.model.save("ppo_multi_vehicle_best")
                                if hasattr(self.training_env, 'save'):
                                    self.training_env.save("ppo_multi_vehicle_best_vec_normalize.pkl")
                                if self.verbose > 0:
                                    print(f"\n🌟 新最佳! 得分:{window_score:.4f} 成功:{success_rate:.1%} 碰撞:{collision_rate:.1%} 超时:{timeout_rate:.1%} (W误:{w_miss_rate:.1%} 倒卡:{rev_stuck_rate:.1%} 全局:{global_limit_rate:.1%})")

        # 改为按步数保存 (每1000步算一个“回合周期”)
        # 只有当：1. 到了保存时间 AND 2. 场景里有车且车跑了一段距离 时，才保存
        if self.num_timesteps - self.last_save_step >= self.save_freq_steps:
            try:
                # 检查第0个环境是否有"正在运行且有轨迹"的活跃车辆
                has_running = self.training_env.get_attr("has_running_agents", indices=[0])[0]
                
                if has_running:
                    self.last_save_step = self.num_timesteps
                    # 伪造一个 episode id 用于命名，或者直接用步数
                    virtual_episode = int(self.num_timesteps // self.save_freq_steps)
                    save_path = os.path.join(self.save_dir, f"multi_episode_{virtual_episode}.png")
                    
                    # 只从第 0 个进程的环境中保存图片
                    self.training_env.env_method("save_training_frame", save_path=save_path, indices=[0])
                    if self.verbose > 0:
                        current_succ = sum(self.recent_successes)/len(self.recent_successes) if self.recent_successes else 0
                        current_coll = sum(self.recent_collisions)/len(self.recent_collisions) if self.recent_collisions else 0
                        current_time = sum(self.recent_timeouts)/len(self.recent_timeouts) if self.recent_timeouts else 0
                        w_m = sum(self.recent_w_misses)/len(self.recent_w_misses) if self.recent_w_misses else 0
                        r_s = sum(self.recent_rev_stucks)/len(self.recent_rev_stucks) if self.recent_rev_stucks else 0
                        g_l = sum(self.recent_global_limits)/len(self.recent_global_limits) if self.recent_global_limits else 0
                        print(f"步骤 {self.num_timesteps}: 成功:{current_succ:.1%} 碰撞:{current_coll:.1%} 超时:{current_time:.1%} (W误:{w_m:.1%} 倒卡:{r_s:.1%} 全局:{g_l:.1%})")
                # else: pass # 没跑开就不存，等自增的下一步再检查
            except Exception as e:
                print(f"Error saving image: {e}")
                
        return True

class MultiAgentEvalCallback(BaseCallback):
    """
    定期在一个纯净的无探索噪音环境中评估模型表现，并保存真正的 Best Model。
    打分公式重构为：成功率 - 5.0 * 碰撞率 - 0.5 * 超时率 (极度厌恶碰撞)
    """
    def __init__(self, eval_env, eval_freq=100_000, n_eval_episodes=10, best_model_save_path="ppo_multi_vehicle_best", verbose=1):
        super(MultiAgentEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.best_score = -float('inf')
        
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_eval_episodes > 0:
            if self.num_timesteps % self.eval_freq == 0:
                self.evaluate_and_save()
        return True
        
    def evaluate_and_save(self):
        if self.verbose > 0:
            print(f"\n--- 暂停训练，开始阶段性纯净评估 (Step {self.num_timesteps}) ---")
        
        successes, collisions, timeouts = 0, 0, 0
        total_dones = 0
        
        # 同步归一化状态 (安全获取 VecNormalize)
        from stable_baselines3.common.vec_env import sync_envs_normalization
        sync_envs_normalization(self.training_env, self.eval_env)
            
        obs = self.eval_env.reset()
        episodes_completed = 0
        
        while episodes_completed < self.n_eval_episodes:
            # Deterministic=True 代表无探索噪音
            action, _states = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = self.eval_env.step(action)
            
            for i, done in enumerate(dones):
                if done:
                    # 获取 info (注意 VecEnv 包装后的 info 结构)
                    # SubprocMultiAgentVecEnv 返回的是 dict
                    info = infos[i] if isinstance(infos, list) else infos
                    
                    if info.get('success', False): successes += 1
                    elif info.get('collision', False): collisions += 1
                    else: timeouts += 1
                    
                    total_dones += 1
                    if total_dones >= self.eval_env.unwrapped.max_agents_per_env: # 一个 episode 完成
                        episodes_completed += 1
                        total_dones = 0
                        break # 重置外层环境

        # 计算指标
        total = successes + collisions + timeouts
        if total == 0: return

        succ_rate = successes / total
        coll_rate = collisions / total
        time_rate = timeouts / total
        
        # 新打分规则：回归 1.0 碰撞惩罚
        score = succ_rate - 1.0 * coll_rate - 0.5 * time_rate
        
        if self.verbose > 0:
            print(f"评估完成! [样本数: {total}] -> 成功: {succ_rate:.1%}, 碰撞: {coll_rate:.1%}, 超时: {time_rate:.1%}")
            print(f"当前得分: {score:.4f} | 历史最佳得分: {self.best_score:.4f}")
            
        self.logger.record("eval/success_rate", succ_rate)
        self.logger.record("eval/collision_rate", coll_rate)
        self.logger.record("eval/score", score)
        
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.best_model_save_path)
            if self.verbose > 0:
                print(f"🎉 保存了新的最佳模型到 {self.best_model_save_path}.zip!\n")


def make_env(spawn_interval=30):
    def _init():
        return MultiVehicleLiDAR2DEnv(max_agents=7, spawn_interval=spawn_interval)
    return _init

def train_multi():
    # 使用 12 个核心进行并行采样
    # 创建 14 个并行环境工厂
    n_procs = 14
    print(f"Initializing Multi-Vehicle MAPPO Training with {n_procs} processes...")
    
    # 设置随机种子 (新增)
    seed = 42
    set_global_seeds(seed)
    
    # 传递种子给环境
    env_fns = [make_env(spawn_interval=30) for _ in range(n_procs)]
    # SubprocVecEnv 会自动处理种子递增：seed, seed+1, ...
    env = SubprocMultiAgentVecEnv(env_fns)
    env.seed(seed)
    
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.)
    
    # 模型路径
    model_path = "ppo_multi_vehicle"
    stats_path = "vec_normalize.pkl"
    
    # 初始化 PPO
    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing model: {model_path} (Forced to CPU)")
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env)
            print("Loaded normalization stats.")
        model = PPO.load(model_path, env=env, device="cpu", custom_objects={"policy_class": MAPPOPolicy})
        model.learning_rate = 1e-4
    else:
        print("Creating new model with MAPPO (Centralized Critic) architecture...")
        policy_kwargs = dict(
            features_extractor_class=MAPPOFeatureExtractor,
            features_extractor_kwargs=dict(),
            activation_fn=nn.Tanh,
            ortho_init=True
        )
        model = PPO(
            MAPPOPolicy,
            env,
            verbose=1,
            learning_rate=linear_schedule(1e-4),
            gamma=0.99,                       # 看得更远，减少因短视导致的超时
            n_steps=2048,
            batch_size=1024,
            n_epochs=10,
            clip_range=0.1,                   # 允许更大的策略更新幅度 (Standard PPO)
            target_kl=0.015,                   # 稍微放宽 Early Stopping
            ent_coef=0.01,                    # 初始值，由 Callback 负责线性衰减
            policy_kwargs=policy_kwargs,
            tensorboard_log="./ppo_multi_tensorboard/",
            device="cpu"
        )
    
    image_callback = EpisodeImageCallback(save_dir="multi_training_images", save_freq_steps=15000, verbose=1)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000, 
        save_path='./logs/',
        name_prefix='ppo_multi_vehicle'
    )
    
    print(f"Starting training (14 Cores / 30M steps) with Level 2 stability settings...")
    model.learn(
        total_timesteps=30_000_000, 
        callback=[checkpoint_callback, image_callback]
    )
    
    model.save(model_path)
    env.save(stats_path)
    print(f"Model and stats saved to {model_path} and {stats_path}")

if __name__ == "__main__":
    train_multi()
