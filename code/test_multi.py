
import os
import numpy as np
import imageio
from stable_baselines3 import PPO
from multi_env import MultiVehicleLiDAR2DEnv, MultiAgentVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import hashlib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
except Exception:
    pass
matplotlib.rcParams['axes.unicode_minus'] = False

# 指标标签
LABELS = [
    "Success Rate",
    "Safety (1 - Collision)",
    "Efficiency (1 / Time)",
    "Avg Velocity",
    "Safety Margin",
    "Smoothness",
    "Reliability (1-Timeout)"
]

def plot_radar_chart(data, labels, output_path="evaluation_radar.png"):
    """
    绘制雷达图。
    :param data: 字典, key 为方法名, value 为指标列表 (长度必须和 labels 一致)
    :param labels: 指标标签列表
    :param output_path: 输出图像路径
    """
    num_vars = len(labels)
    
    # 计算角度 (等分圆周)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
    
    for idx, (method, values) in enumerate(data.items()):
        values = values + values[:1]  # 闭合
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # 设置刻度
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=9)
    
    # 获取第一个方法的数据（通常是本方案）
    first_method_name = list(data.keys())[0]
    first_method_values = data[first_method_name]
    
    # 构造带数值的标签，例如: "Success Rate\n(95)"
    annotated_labels = []
    for label, val in zip(labels, first_method_values):
        annotated_labels.append(f"{label}\n({val:.1f})")

    # 设置标签 (增加 pad 参数防止重叠)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(annotated_labels, fontsize=11, fontweight='bold')
    ax.tick_params(pad=20) # 关键：增加 padding 让文字远离图形
    
    # 图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plt.title("Multi-Dimensional Performance Comparison", fontsize=14, fontweight='bold', y=1.08)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Radar chart saved to: {output_path}")
    # plt.show() # 在服务器或批量运行时通常不需要 show

def evaluate_multi_agent(model_path="ppo_multi_vehicle_best", n_episodes=30, video_path="multi_eval_3_episodes.mp4"):
    import random
    import numpy as np
    import torch
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("Initializing environment...")
    # 恢复为 7 车配置
    raw_env = MultiVehicleLiDAR2DEnv(max_agents=7, spawn_interval=30)
    
    # 封装为多智能体 VecEnv (关键修正)
    env_vec = MultiAgentVecEnv(raw_env)
    
    stats_path = "vec_normalize.pkl"
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}...")
        env_vec = VecNormalize.load(stats_path, env_vec)
        # 评估时不更新统计信息
        env_vec.training = False
        env_vec.norm_reward = False
    
    # 需要在加载模型前定义 MAPPO 相关的类，以便 PPO.load 能识别自定义对象
    # (如果 train_multi.py 就在同一目录下，可以直接 import，或者在这里临时定义)
    from train_multi import MAPPOPolicy, MAPPOFeatureExtractor 
    
    actual_model_path = model_path
    if not os.path.exists(actual_model_path) and os.path.exists(actual_model_path + ".zip"):
        actual_model_path += ".zip"
        
    if not os.path.exists(actual_model_path):
        print(f"Error: Model file {actual_model_path} not found.")
        return
    
    print(f"Loading model from {actual_model_path}...")
    model = PPO.load(actual_model_path, env=env_vec, device="cpu", custom_objects={"policy_class": MAPPOPolicy})
    
    # 注意：后续代码使用的是 env，由于 VecNormalize 是对 env 的包装，
    # 我们需要确保后续渲染和步进逻辑能通过 env_vec 访问到原始环境或正确处理。
    # 简单起见，在 evaluate 循环内使用 env_vec。
    
    env = raw_env # 保留原始环境引用用于 render 等非 obs 相关的调用
    
    frames = []
    hashes = []

    # --- 统计数据初始化 ---
    total_completed = 0
    total_success = 0
    total_collision = 0
    total_timeout = 0
    
    # 细分超时类型
    timeout_w_miss = 0
    timeout_rev_stuck = 0
    timeout_global = 0
    
    # 效率指标 trackers
    service_times = []      # 完成任务的步数
    velocities = []         # 所有活跃时刻的瞬时速度
    detour_rates = []       # 路径偏差率 (实际/直线)
    
    # 安全/质量指标 trackers
    global_min_dist = float('inf')
    smoothness_scores = []  # 加速度/转向变化率
    
    # 容量指标 trackers
    all_active_counts = []  # 记录每一帧的活跃车辆数

    # 状态追踪 (用于计算上述指标)
    # key: agent_idx
    agent_start_steps = {}  # 任务开始时间
    agent_odometers = {}    # 累计行驶里程
    agent_last_pos = {}     # 上一步位置
    agent_last_action = {}  # 上一步动作 (v, steer)
    
    # 固定参数 (从环境获取)
    ENTRY_POS = env.entry_pos
    EXIT_POS = env.exit_pos
    
    for ep in range(n_episodes):
        print(f"\n>>> Starting Episode {ep + 1}/{n_episodes}")
        obs = env_vec.reset() 
        step = 0
        episode_active_counts = []
        
        # 重置本局状态追踪 (防止跨局污染)
        current_active_mask = np.zeros(env.max_agents, dtype=bool)
        
        while step < 1500:
            pre_step_positions = env.positions.copy()
            
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env_vec.step(action)

            new_active_mask = env.active_mask.copy()
            
            for i in range(env.max_agents):
                if new_active_mask[i] and not current_active_mask[i]:
                    agent_start_steps[i] = step
                    agent_odometers[i] = 0.0
                    agent_last_pos[i] = env.positions[i].copy()
                    agent_last_action[i] = np.array([0.0, 0.0]) 
            
            current_active_mask = new_active_mask
            
            active_indices = [i for i, a in enumerate(env.active_mask) if a]
            episode_active_counts.append(len(active_indices))
            
            if len(active_indices) >= 2:
                for i in range(len(active_indices)):
                    for j in range(i + 1, len(active_indices)):
                        idx1, idx2 = active_indices[i], active_indices[j]
                        dist = np.linalg.norm(env.positions[idx1] - env.positions[idx2])
                        if dist < global_min_dist:
                            global_min_dist = dist
            
            for i in active_indices:
                curr_pos = env.positions[i]
                prev_pos = pre_step_positions[i] 
                step_dist = np.linalg.norm(curr_pos - prev_pos)
                
                agent_odometers[i] = agent_odometers.get(i, 0.0) + step_dist
                velocities.append(step_dist) 
                
                curr_act = action[i]
                last_act = agent_last_action.get(i, curr_act) 
                smooth_cost = np.abs(curr_act[0] - last_act[0]) + np.abs(curr_act[1] - last_act[1])
                smoothness_scores.append(smooth_cost)
                agent_last_action[i] = curr_act 

            for i, done in enumerate(dones):
                if done:
                    # 避免对已被处理的非活跃槽位重复计数 (因为VecEnv对所有done都返回True)
                    # 不过infos[i]里的状态是由环境真实抛出的
                    is_succ = infos[i].get('success', False)
                    is_coll = infos[i].get('collision', False)
                    is_time = infos[i].get('timeout', False)
                    reason = infos[i].get('timeout_reason', '')
                    
                    if not is_succ and not is_coll and not is_time:
                        continue # 过滤幽灵回合
                        
                    total_completed += 1
                    
                    if is_succ:
                        total_success += 1
                        print(f"  [Vehicle #{i}] Reached Target!")
                        if i in agent_start_steps:
                            duration = step - agent_start_steps[i]
                            service_times.append(duration)
                            actual_dist = agent_odometers.get(i, 0.0)
                            target_site = env.target_sites[i]
                            if target_site:
                                target_pos = target_site['pos']
                                optimal_dist = np.linalg.norm(ENTRY_POS - target_pos) + \
                                               np.linalg.norm(target_pos - EXIT_POS)
                                if optimal_dist > 1e-6:
                                    ratio = actual_dist / optimal_dist
                                    detour_rates.append(ratio)
                            
                            agent_start_steps[i] = step
                            agent_odometers[i] = 0.0
                            agent_last_pos[i] = env.positions[i].copy()
                            
                    elif is_coll:
                        total_collision += 1
                        print(f"  [Vehicle #{i}] Collided!")
                        
                    elif is_time:
                        total_timeout += 1
                        if reason == 'w_point_miss':
                            timeout_w_miss += 1
                            print(f"  [Vehicle #{i}] Times Out (W-Point Miss)!")
                        elif reason == 'reversing_stuck':
                            timeout_rev_stuck += 1
                            print(f"  [Vehicle #{i}] Times Out (Reversing Stuck)!")
                        else:
                            timeout_global += 1
                            print(f"  [Vehicle #{i}] Times Out (Global Limit)!")
            
            if step % 50 == 0:
                active_ids = [i for i, a in enumerate(env.active_mask) if a]
                # 统计安全干预次数
                interventions = [infos[i].get('safety_intervention', False) for i in active_ids]
                int_str = f" | Shield: {sum(interventions)}" if any(interventions) else ""
                
                pos_str = " | ".join([f"Veh#{i}:{env.positions[i]}".replace('\n','') for i in active_ids])
                print(f"  [Step {step:04d}] Active: {len(active_ids)} {int_str} | {pos_str}")
            
            # 记录帧 (降低采样率，每 8 步采一帧，减轻 GIF 压力)
            if step % 8 == 0:
                frame = env.render_frame()
                curr_hash = hashlib.md5(frame.tobytes()).hexdigest()
                hashes.append(curr_hash)
                frames.append(frame)
                
                # 保留少量的静态证据图
                if len(frames) % 300 == 1:
                    debug_img_path = f"debug_frame_{len(frames)}.png"
                    Image.fromarray(frame).save(debug_img_path)
            
            step += 1
            
        print(f"Episode {ep + 1} finished.")
        
        # 记录本回合的活跃数统计
        all_active_counts.extend(episode_active_counts)

    
    # 统计信息
    unique_hashes = len(set(hashes))
    print(f"\n[Summary] Collected {len(frames)} frames. Unique: {unique_hashes}")

    if total_completed > 0:
        success_rate = total_success / total_completed
        collision_rate = total_collision / total_completed
        timeout_rate = total_timeout / total_completed
        
        avg_service_time = np.mean(service_times) if service_times else 0.0
        avg_velocity = np.mean(velocities) if velocities else 0.0
        avg_detour = np.mean(detour_rates) * 100 if detour_rates else 0.0 # percentage
        avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0.0
        min_margin = global_min_dist if global_min_dist != float('inf') else 0.0
        
        print("\n" + "="*50)
        print(f"📊 Comprehensive Evaluation Results ({n_episodes} Episodes)")
        print(f"Tasks Completed:      {total_completed}")
        print(f"Success Rate:         {success_rate:.2%}")
        print(f"Collision Rate:       {collision_rate:.2%}")
        print(f"Timeout Rate (Total): {timeout_rate:.2%}")
        
        # 详细倒车超时分类比率 (相对于所有任务)
        w_miss_rate = timeout_w_miss / total_completed if total_completed > 0 else 0
        rev_stuck_rate = timeout_rev_stuck / total_completed if total_completed > 0 else 0
        global_rate = timeout_global / total_completed if total_completed > 0 else 0
        print(f"  ├─ W-Point Miss:    {w_miss_rate:.2%}")
        print(f"  ├─ Reversing Stuck: {rev_stuck_rate:.2%}")
        print(f"  └─ Global Limit:    {global_rate:.2%}")
        
        print("-" * 50)
        
        # 计算容量饱和度指标
        avg_occupancy = np.mean(all_active_counts) if all_active_counts else 0.0
        max_capacity = env.max_agents
        saturation_rate = (np.sum(np.array(all_active_counts) == max_capacity) / len(all_active_counts)) * 100 if all_active_counts else 0.0
        
        print("Capacity Metrics:")
        print(f"Avg Occupancy:        {avg_occupancy:.2f} / {max_capacity} agents")
        print(f"Saturation Rate:      {saturation_rate:.2f}% (Time at Full Capacity)")
        print("-" * 50)
        print("Efficiency Metrics:")
        print(f"Avg Service Time:     {avg_service_time:.1f} steps")
        print(f"Avg Velocity:         {avg_velocity:.4f} m/step") 
        print(f"Avg Detour Rate:      {avg_detour:.1f}% (100% is perfect)")
        print("-" * 50)
        print("Safety & Quality Metrics:")
        print(f"Min Safety Margin:    {min_margin:.3f} m")
        print(f"Motion Smoothness:    {avg_smoothness:.4f} (Lower is better)")
        print("="*50 + "\n")
        
        # --- 自动生成雷达图 ---
        try:
            # 1. 数据归一化 (将各指标映射到 0-100，越大越好)
            # 设定一些基准值 (Based on empirical values)
            norm_success = success_rate * 100
            norm_safety = (1.0 - collision_rate) * 100
            
            # 效率: 假设 200步是满分(100分), 1000步是0分
            # Formula: 100 - (time - 200) / 8
            norm_efficiency = np.clip(100 - (avg_service_time - 200) / 8, 0, 100)
            
            # 速度: 假设 0.5 m/s 是满分, 0 是 0分
            norm_velocity = np.clip((avg_velocity / 0.5) * 100, 0, 100)
            
            # 间距: 假设 2.0m 是满分 (安全), 0 m 是 0分
            norm_margin = np.clip((min_margin / 2.0) * 100, 0, 100)
            
            # 平滑度: 假设 0 是满分, 0.2 是 0分
            # Formula: (0.2 - val) / 0.2 * 100
            norm_smoothness = np.clip((0.2 - avg_smoothness) / 0.2 * 100, 0, 100)
            
            # 超时率: 假设 0 是满分, 100% (1.0) 是 0分
            norm_timeout = np.clip((1.0 - timeout_rate) * 100, 0, 100)

            radar_data = {
                "Our Method": [norm_success, norm_safety, norm_efficiency, norm_velocity, norm_margin, norm_smoothness, norm_timeout],
                # 添加一个虚拟的 Baseline 用于对比效果 (可以注释掉)
                "Baseline (Random)": [30, 40, 30, 40, 20, 30, 50] 
            }
            
            print("Generating Radar Chart...")
            plot_radar_chart(radar_data, LABELS, output_path="evaluation_radar.png")
            
        except Exception as e:
            print(f"Radar chart generation failed: {e}")

    else:
        print("\n⚠️ No agents completed their tasks during evaluation.")

    # 尝试保存视频或 GIF
    try:
        # 如果 ffmpeg 可用，尝试保存 MP4
        import imageio
        print(f"Attempting to save MP4 to {video_path}...")
        imageio.mimsave(video_path, frames, fps=15, macro_block_size=None)
        print("Successfully saved MP4.")
    except Exception as e:
        # 回退到使用 PIL 保存 GIF
        gif_out = video_path.replace(".mp4", ".gif")
        print(f"MP4 failed ({e}), saving via PIL to {gif_out}...")
        
        # 关键：将 RGB 转换为调色板模式 (P mode)，这是 GIF 动画的正确格式
        imgs_pil = [Image.fromarray(f).convert('P', palette=Image.ADAPTIVE, colors=256) for f in frames]
        
        imgs_pil[0].save(
            gif_out, 
            save_all=True, 
            append_images=imgs_pil[1:], 
            duration=66,  # 约 15fps
            loop=0,       # 无限循环
            disposal=2    # 每帧完全替换上一帧
        )
        print(f"Successfully saved GIF to {gif_out} ({len(frames)} frames)")

if __name__ == "__main__":
    evaluate_multi_agent()
