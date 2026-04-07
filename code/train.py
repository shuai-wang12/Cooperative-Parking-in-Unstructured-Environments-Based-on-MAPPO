import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from env import CustomLiDAR2DEnv
import os

class EpisodeImageCallback(BaseCallback):
    """
    自定义回调：每隔一定数量的回合保存环境最后一帧的图像。
    兼容多进程环境。
    """
    def __init__(self, save_dir, save_freq_episodes=200, verbose=0):
        super(EpisodeImageCallback, self).__init__(verbose)
        self.save_dir = save_dir
        self.save_freq_episodes = save_freq_episodes
        self.episode_count = 0
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # 多进程环境下，dones 是一个数组，统计所有完成的回合
        dones = self.locals.get("dones")
        if dones is not None:
            # 统计本次有多少个环境完成了回合
            n_done = sum(dones)
            self.episode_count += n_done
            
            # 如果任意环境完成了回合，且达到保存频率
            if n_done > 0 and self.episode_count % self.save_freq_episodes < n_done:
                save_path = os.path.join(self.save_dir, f"episode_{self.episode_count}.png")
                # 只从第一个环境保存图像（indices=[0]）
                self.training_env.env_method("save_training_frame", save_path=save_path, indices=[0])
                if self.verbose > 0:
                    print(f"已保存第 {self.episode_count} 回合的渲染图至 {save_path}")
        return True

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def train():
    # 使用 8 个并行环境（根据 CPU 核数调整，通常设为核数 - 2）
    n_envs = 12
    
    # 先用单个环境验证规范
    print("正在检查环境规范...")
    single_env = CustomLiDAR2DEnv()
    check_env(single_env)
    single_env.close()
    
    # 创建多进程并行环境
    print(f"创建 {n_envs} 个并行环境...")
    env = make_vec_env(CustomLiDAR2DEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    
    # 初始化模型（优化超参数以稳定训练）
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=1e-4,          # 原默认 3e-4，降低以稳定训练
        batch_size=256,              # 原默认 64，增大以减少噪声
        clip_range=0.1,              # 原默认 0.2，限制策略更新幅度
        ent_coef=0.01,               # 原默认 0.0，增加探索
        tensorboard_log="./ppo_tensorboard/"
    )
    
    # 设置图像保存路径和回调（注意：多进程下 episode_count 是所有环境累加的）
    image_callback = EpisodeImageCallback(save_dir="training_images", save_freq_episodes=1000, verbose=1)
    
    # 训练
    print("开始多进程并行训练...")
    model.learn(total_timesteps=1000000000, callback=image_callback)
    
    # 保存模型
    model_path = "ppo_lidar_nav"
    model.save(model_path)
    print(f"模型已保存至: {model_path}")

def test():
    env = CustomLiDAR2DEnv()
    model = PPO.load("ppo_lidar_nav")
    
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(250):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"测试结束，总奖励: {total_reward}, 状态: {'成功' if reward > 10 else '失败/超时'}")
            break

if __name__ == "__main__":
    train()
    # test()
