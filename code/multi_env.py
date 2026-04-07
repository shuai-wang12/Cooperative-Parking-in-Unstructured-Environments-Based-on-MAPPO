import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Avoid OpenMP duplicate library error
import gymnasium as gym
from gymnasium import spaces
import numpy as np
# Remove global matplotlib import to avoid heap corruption in subprocesses
import random
from stable_baselines3.common.vec_env import VecEnv

# Helper to avoid circular imports or missing definitions
from env import CustomLiDAR2DEnv

class MultiVehicleLiDAR2DEnv(gym.Env):
    """
    多车动态进出环境。
    支持最多 max_agents 辆车同时在场。
    车辆在入口点定时产生，完成任务后从出口点离场。
    """
    def __init__(self, max_agents=7, spawn_interval=100):
        super(MultiVehicleLiDAR2DEnv, self).__init__()
        self.max_agents = max_agents
        self.spawn_interval = spawn_interval
        
        # 障碍物线段 (硬编码以确保与 env.py 完全一致)
        self.obstacle_lines = [
            [np.array([2.0, 10.0]), np.array([0.0, 4.0])], [np.array([0.0, 4.0]), np.array([0.0, 3.0])],
            [np.array([0.0, 3.0]), np.array([1.0, 1.0])], [np.array([1.0, 1.0]), np.array([3.0, 0.0])],
            [np.array([3.0, 0.0]), np.array([7.0, 0.0])], [np.array([7.0, 0.0]), np.array([10.0, 1.0])],
            [np.array([10.0, 1.0]), np.array([10.0, 5.0])], [np.array([10.0, 5.0]), np.array([6.0, 6.0])],
            [np.array([6.0, 6.0]), np.array([4.0, 7.0])], [np.array([4.0, 7.0]), np.array([5.0, 10.0])],
            [np.array([5.0, 10.0]), np.array([2.0, 10.0])], [np.array([3.5, 10.0]), np.array([2.5, 7.0])]
        ]
        
        self.boundary_coords = np.array([
            [2.0, 10.0], [0.0, 4.0], [0.0, 3.0], [1.0, 1.0], [3.0, 0.0],
            [7.0, 0.0], [10.0, 1.0], [10.0, 5.0], [6.0, 6.0], [4.0, 7.0], [5.0, 10.0]
        ])

        self.obs_segments = np.array(self.obstacle_lines)
        
        # 借用单车环境的站点采样逻辑 (仅调用一次)
        base_env = CustomLiDAR2DEnv()
        self.dumping_sites = base_env.dumping_sites
        base_env.close()
        
        # 车辆参数
        self.agent_length = 0.5
        self.agent_width = 0.2
        self.wheelbase = 0.3
        self.min_turning_radius = 1.25
        self.max_steer = np.arctan(self.wheelbase / self.min_turning_radius)
        self.lidar_range = 5.0
        self.n_lidar_rays = 32 # 增加射线密度，减少盲区
        self.decision_dist = 4.0
        
        # 入口和出口
        self.entry_pos = np.array([2.08, 8.0], dtype=np.float32)
        self.entry_yaw = -1.901295
        self.exit_pos = np.array([3.58, 8.0], dtype=np.float32)
        self.exit_yaw = self.entry_yaw + np.pi
        
        # 状态追踪 (数组形式，索引为 agent_id)
        self.active_mask = np.zeros(max_agents, dtype=bool)
        self.positions = np.zeros((max_agents, 2), dtype=np.float32)
        self.yaws = np.zeros(max_agents, dtype=np.float32)
        self.phases = ["inactive"] * max_agents  # inactive, forward_rl, reversing, exit
        self.target_sites = [None] * max_agents
        self.locked_w_poses = [None] * max_agents
        self.bezier_wps = [None] * max_agents
        self.goal_switched = np.zeros(max_agents, dtype=bool)
        self.arc_progress = np.zeros(max_agents, dtype=np.float32)
        self.prev_actions = np.zeros((max_agents, 2), dtype=np.float32)
        self.prev_potential_costs = np.zeros(max_agents, dtype=np.float32)
        self.traj_histories = [[] for _ in range(max_agents)]
        
        # "错过即失败" 追踪变量
        self.min_dist_to_w = np.full(max_agents, np.inf, dtype=np.float32)
        self.approach_started = np.zeros(max_agents, dtype=bool)
        self.overshoot_steps = np.zeros(max_agents, dtype=np.int32)
        
        # "静止判定" 追踪变量 (W点低速帧数)
        self.low_speed_frames = np.zeros(max_agents, dtype=np.int32)
        
        # "W点超时" 追踪变量 (在W点附近徘徊的步数)
        self.near_w_steps = np.zeros(max_agents, dtype=np.int32)
        
        # "倒车死锁" 追踪变量 (Stage 2 被其他车堵住的步数)
        self.reversing_stuck_steps = np.zeros(max_agents, dtype=np.int32)
        
        # "让行优先级" 追踪变量 (记录 goal_switched 触发的时间戳)
        self.goal_switch_step = np.zeros(max_agents, dtype=np.int32)
        
        self.steps_since_last_spawn = 0
        self.current_step = 0
        self.max_steps = 1500 # 动态环境通常步骤较长
        
        # 动作空间：[速度, 转向角] (由于是共享策略，动作空间定义为单车的)
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.max_steer]), 
            high=np.array([0.5, self.max_steer]), 
            dtype=np.float32
        )
        
        # 观测空间：32(lidar) + 2(rel_pos) + 2(rel_yaw) + 2(prev_action) + 2(phase_indicator) = 40
        # + (max_agents-1)*7 (其他车辆: rel_x, rel_y, sin_yaw, cos_yaw, norm_v, norm_steer, phase_priority)
        # 观测空间更新为 Dict 以支持 MAPPO (CTDE)
        self.self_obs_dim = self.n_lidar_rays + 2 + 2 + 2 + 2 
        self.other_obs_dim = (max_agents - 1) * 7
        self.local_obs_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(self.self_obs_dim + self.other_obs_dim,), 
            dtype=np.float32
        )
        
        # 全局状态空间 (Critic 用): 5车 * (x,y,sin,cos,phase_idx,potential,goal_switched,is_active) + 8站点
        # phase_idx: 0:inactive, 1:forward, 2:reversing, 3:exit
        self.global_state_dim = max_agents * 8 + len(self.dumping_sites)
        self.global_state_space = spaces.Box(
            low=-5.0, high=10.0,
            shape=(self.global_state_dim,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            "obs": self.local_obs_space,
            "state": self.global_state_space
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.steps_since_last_spawn = self.spawn_interval 
        
        self.active_mask.fill(False)
        self.positions.fill(0)
        self.yaws.fill(0)
        self.phases = ["inactive"] * self.max_agents
        self.target_sites = [None] * self.max_agents
        self.locked_w_poses = [None] * self.max_agents
        self.bezier_wps = [None] * self.max_agents
        self.goal_switched.fill(False)
        self.arc_progress.fill(0)
        self.prev_actions.fill(0)
        self.prev_potential_costs.fill(0)
        self.traj_histories = [[] for _ in range(self.max_agents)]
        
        self.min_dist_to_w.fill(np.inf)
        self.approach_started.fill(False)
        self.overshoot_steps.fill(0)
        self.low_speed_frames.fill(0)
        self.near_w_steps.fill(0)
        self.reversing_stuck_steps.fill(0)
        self.goal_switch_step.fill(0)
        
        self._try_spawn()
        
        return self._get_all_obs_dict(), {}

    def _get_all_obs_dict(self):
        # 返回 SB3 格式的 Dict Obs
        local_obs = []
        for i in range(self.max_agents):
            local_obs.append(self._get_obs(i))
        
        global_state = self._get_global_state()
        # 对于所有 Agent，Global State 是共享的
        # 注意：Agent-Specific State 在 MLP 架构下无效（需要 Attention），已回滚
        global_states = np.tile(global_state, (self.max_agents, 1))
        
        return {
            "obs": np.stack(local_obs),
            "state": global_states
        }

    def _get_global_state_components(self):
        """返回结构化的全局状态组件，用于 Agent-Specific State 重排"""
        agent_feats = []
        phase_map = {"inactive": 0, "forward_rl": 1, "reversing": 2, "exit": 3}
        
        for i in range(self.max_agents):
            pos = self.positions[i] / 10.0 
            yaw = self.yaws[i]
            phase_idx = phase_map.get(self.phases[i], 0) / 3.0
            potential = self.prev_potential_costs[i] / 50.0
            gs = 1.0 if self.goal_switched[i] else 0.0
            is_active = 1.0 if self.active_mask[i] else 0.0
            
            agent_feats.append(np.array([pos[0], pos[1], np.sin(yaw), np.cos(yaw), 
                                          phase_idx, potential, gs, is_active], dtype=np.float32))
        
        # 站点占用状态
        site_occupancy = []
        used_target_poses = [self.target_sites[i]['pos'] for i in range(self.max_agents)
                            if self.active_mask[i] and self.target_sites[i] is not None]
        for site in self.dumping_sites:
            occupied = 0.0
            for up in used_target_poses:
                if np.array_equal(site['pos'], up):
                    occupied = 1.0; break
            site_occupancy.append(occupied)
            
        return agent_feats, np.array(site_occupancy, dtype=np.float32)

    def _get_global_state(self):
        """ 为 MAPPO Critic 提供全局一致的 Joint State (向后兼容) """
        agent_feats, site_occupancy = self._get_global_state_components()
        return np.concatenate(agent_feats + [site_occupancy]).astype(np.float32)

    def _get_obs(self, idx):
        if not self.active_mask[idx]:
            return np.zeros(self.local_obs_space.shape, dtype=np.float32)
        
        # 1. LiDAR (考虑其他车辆作为动态障碍物)
        lidar = self._get_lidar_readings(idx) / self.lidar_range
        
        # 2. 目标信息
        if self.phases[idx] == "exit":
            target_pos = self.exit_pos
            target_yaw = self.exit_yaw
        else:
            target_pos = self.locked_w_poses[idx]['pos']
            target_yaw = self.locked_w_poses[idx]['yaw']
            
        dx, dy = target_pos[0] - self.positions[idx][0], target_pos[1] - self.positions[idx][1]
        c, s = np.cos(self.yaws[idx]), np.sin(self.yaws[idx])
        
        # 恢复原始相对位置 (使用 10.0 的缩放，与其它车辆的缩放比例一致)
        rx, ry = (dx*c + dy*s) / 10.0, (-dx*s + dy*c) / 10.0
        dyaw = (target_yaw - self.yaws[idx] + np.pi) % (2 * np.pi) - np.pi
        
        # 动作归一化
        norm_v = self.prev_actions[idx][0] / 0.5
        norm_steer = self.prev_actions[idx][1] / self.max_steer
        
        # 3. 阶段指示器 (one-hot: [is_forward_rl, is_exit])
        is_forward = 1.0 if self.phases[idx] == "forward_rl" else 0.0
        is_exit = 1.0 if self.phases[idx] == "exit" else 0.0
        
        self_obs = np.concatenate([
            lidar, 
            np.array([rx, ry], dtype=np.float32), 
            np.array([np.sin(dyaw), np.cos(dyaw)], dtype=np.float32),
            np.array([norm_v, norm_steer], dtype=np.float32),
            np.array([is_forward, is_exit], dtype=np.float32)  # 新增：阶段指示器
        ])
        
        # 4. 其他车辆信息 (按距离排序，包含速度信息)
        others_data = []
        
        for j in range(self.max_agents):
            if idx == j: continue
            if self.active_mask[j]:
                dx_j, dy_j = self.positions[j][0] - self.positions[idx][0], self.positions[j][1] - self.positions[idx][1]
                dist_j = dx_j**2 + dy_j**2
                
                # 转换到局部坐标系
                rx_j, ry_j = (dx_j*c + dy_j*s) / 10.0, (-dx_j*s + dy_j*c) / 10.0
                dyaw_j = (self.yaws[j] - self.yaws[idx] + np.pi) % (2 * np.pi) - np.pi
                
                # 新增：对方的速度和转向 (归一化)
                v_j = self.prev_actions[j][0] / 0.5
                steer_j = self.prev_actions[j][1] / self.max_steer
                
                # 新增：对方的让行优先级 (归一化到 0~1)
                phase_priority_j = self._get_priority(j)
                
                others_data.append((dist_j, [rx_j, ry_j, np.sin(dyaw_j), np.cos(dyaw_j), v_j, steer_j, phase_priority_j]))
            else:
                pass 
                # 不活跃的车暂时不加入列表，最后统一补零。
                # 或者如果你希望明确表示“无车”，可以不加。
                # 这里的逻辑是：只把“有车”的特征拿来排序，剩下的位置填0。
        
        # 按距离从小到大排序
        others_data.sort(key=lambda x: x[0])
        
        # 展平特征
        others_info = []
        for _, feat in others_data:
            others_info.extend(feat)
            
        # 补齐剩余的空位
        current_len = len(others_info)
        target_len = self.other_obs_dim
        if current_len < target_len:
            others_info.extend([0.0] * (target_len - current_len))
                
        return np.concatenate([self_obs, np.array(others_info, dtype=np.float32)])

    def _get_lidar_readings(self, idx):
        # 组合静态障碍物和动态障碍物（其他车辆）
        dynamic_segments = []
        for j in range(self.max_agents):
            if j == idx or not self.active_mask[j]: continue
            # 这里的 get_rect_corners 需要定义
            corners = self.get_rect_corners(self.positions[j], self.yaws[j], self.agent_length, self.agent_width)
            # 矩形四个边作为线段
            for k in range(4):
                dynamic_segments.append([corners[k], corners[(k+1)%4]])
        
        all_segments = np.concatenate([self.obs_segments, np.array(dynamic_segments)]) if dynamic_segments else self.obs_segments
        
        angles = np.linspace(0, 2*np.pi, self.n_lidar_rays, endpoint=False) + self.yaws[idx]
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        p = self.positions[idx]
        a = all_segments[:,0,:]; b = all_segments[:,1,:]; v = b-a
        dx, dy = dirs[:,0][:,np.newaxis], dirs[:,1][:,np.newaxis]; vx, vy = v[:,0][np.newaxis,:], v[:,1][np.newaxis,:]
        det = -dx*vy + vx*dy; det[np.abs(det)<1e-9] = 1e-9
        rx, ry = a[:,0][np.newaxis,:]-p[0], a[:,1][np.newaxis,:]-p[1]
        t, u = (-rx*vy + vx*ry)/det, (dx*ry - dy*rx)/det
        valid = (t>0) & (u>=0) & (u<=1)
        return np.clip(np.min(np.where(valid, t, self.lidar_range), axis=1), 0.0, self.lidar_range)

    def _get_priority(self, idx):
        """
        计算智能体的让行优先级 (0.0 ~ 1.0)。
        优先级高的车享有路权，优先级低的车应主动避让。
        reversing > forward_rl(已对准,先到) > forward_rl(已对准,后到) > forward_rl(赶路) > exit
        """
        phase = self.phases[idx]
        if phase == "reversing":
            return 1.0
        elif phase == "forward_rl":
            if self.goal_switched[idx]:
                # 先到先得：等待时间越长 → 优先级越高 (趋近 0.8)
                wait = min(self.current_step - self.goal_switch_step[idx], 100)
                return 0.7 + 0.1 * (wait / 100.0)
            else:
                return 0.5
        elif phase == "exit":
            return 0.25
        return 0.0  # inactive

    def get_rect_corners(self, pos, yaw, length, width):
        c, s = np.cos(yaw), np.sin(yaw); rot = np.array([[c, -s], [s, c]])
        half_l, half_w = length / 2.0, width / 2.0
        corners = np.array([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]])
        return corners @ rot.T + pos

    def _try_spawn(self):
        # 末期停止发车：剩余时间不足以完成一趟任务时，不再生成新车
        remaining_steps = self.max_steps - self.current_step
        if remaining_steps < 200:  # 平均 Service Time ~120步，留 200步的安全余量
            return False
        
        # 检查是否有空位
        if np.sum(self.active_mask) >= self.max_agents:
            return False
        
        # 检查起点是否被占用
        for i in range(self.max_agents):
            if self.active_mask[i] and np.linalg.norm(self.positions[i] - self.entry_pos) < 1.0:
                return False
        
        # 寻找空闲索引
        idx = np.where(~self.active_mask)[0][0]
        
        # 初始化车辆
        pos_noise = np.random.uniform(-0.1, 0.1, size=2).astype(np.float32)
        yaw_noise = np.random.uniform(-0.087, 0.087)
        self.positions[idx] = self.entry_pos + pos_noise
        self.yaws[idx] = self.entry_yaw + yaw_noise
        self.active_mask[idx] = True
        self.phases[idx] = "forward_rl"
        self.goal_switched[idx] = False
        self.arc_progress[idx] = 0.0
        self.traj_histories[idx] = [self.positions[idx].copy()]
        
        # 重置"错过即失败"追踪变量
        self.min_dist_to_w[idx] = np.inf
        self.approach_started[idx] = False
        self.overshoot_steps[idx] = 0
        self.low_speed_frames[idx] = 0
        
        # 收集所有已占用的目标排土点和 W 点
        used_target_poses = [self.target_sites[i]['pos'] for i in range(self.max_agents)
                            if self.active_mask[i] and i != idx and self.target_sites[i] is not None]
        used_w_poses = [self.locked_w_poses[i]['pos'] for i in range(self.max_agents) 
                        if self.active_mask[i] and i != idx and self.locked_w_poses[i] is not None]
        
        available_sites = []
        for site in self.dumping_sites:
            # 校验 1: 该站点不能已被其他车辆选为目标 (一车一位)
            is_site_occupied = False
            for used_pos in used_target_poses:
                if np.array_equal(site['pos'], used_pos):
                    is_site_occupied = True
                    break
            if is_site_occupied: continue

            # 校验 2: 一个站点只要有一个 W 点是安全的，就视为可用
            has_safe_w = False
            for w in site['w_poses']:
                is_w_safe = True
                # 检查与其他车辆已锁定 W 点的距离
                for used_w_pos in used_w_poses:
                    if np.linalg.norm(w['pos'] - used_w_pos) < 1.5:
                        is_w_safe = False
                        break
                if not is_w_safe: continue
                
                # 检查与其他排土点（不管是空闲还是占用）的距离
                for other_site in self.dumping_sites:
                    if np.array_equal(other_site['pos'], site['pos']): continue
                    if np.linalg.norm(w['pos'] - other_site['pos']) < 0.6:
                        is_w_safe = False
                        break
                
                if is_w_safe:
                    has_safe_w = True
                    break
            
            if has_safe_w:
                available_sites.append(site)
        
        if not available_sites:
            available_sites = self.dumping_sites  # 回退
        self.target_sites[idx] = random.choice(available_sites)
        
        # 初始化势能
        _, bw, bidx = self._calculate_best_w(idx)
        self.locked_w_poses[idx] = bw
        self.prev_potential_costs[idx] = self._compute_potential(idx)
        
        self.steps_since_last_spawn = 0
        return True

    def _calculate_best_w(self, idx):
        # 专门为多车环境解耦的换向点计算，增加动态冲突规避
        pos = self.positions[idx]; yaw = self.yaws[idx]; site = self.target_sites[idx]
        used_w_poses = [self.locked_w_poses[i]['pos'] for i in range(self.max_agents) 
                        if self.active_mask[i] and i != idx and self.locked_w_poses[i] is not None]
        
        min_c = float('inf'); bw = None; bidx = -1
        
        # 优先选择“未被占用”且“安全”的点
        for i, w in enumerate(site['w_poses']):
            # 冲突检测
            is_occupied = False
            for used_pos in used_w_poses:
                if np.linalg.norm(w['pos'] - used_pos) < 0.6:
                    is_occupied = True
                    break
            if is_occupied: continue
            
            d = np.linalg.norm(pos - w['pos'])
            dx, dy = w['pos'][0] - pos[0], w['pos'][1] - pos[1]
            h_to_p = np.arctan2(dy, dx)
            h_diff = (h_to_p - yaw + np.pi) % (2*np.pi) - np.pi
            if np.abs(h_diff) > np.pi / 2: continue
            entry_align = (h_to_p - w['yaw'] + np.pi) % (2*np.pi) - np.pi
            c = 0.2 * d + 3.0 * np.abs(h_diff) + 6.0 * np.abs(entry_align)
            if c < min_c: min_c = c; bw = w; bidx = i
            
        if bw is None: # fallback: 如果没有理想点，放宽条件（比如包含稍微被占用的点或身后的点）
            for i, w in enumerate(site['w_poses']):
                d = np.linalg.norm(pos - w['pos'])
                dx, dy = w['pos'][0] - pos[0], w['pos'][1] - pos[1]
                h_to_p = np.arctan2(dy, dx)
                h_diff = (h_to_p - yaw + np.pi) % (2*np.pi) - np.pi
                entry_align = (h_to_p - w['yaw'] + np.pi) % (2*np.pi) - np.pi
                c = 0.2 * d + 3.0 * np.abs(h_diff) + 6.0 * np.abs(entry_align)
                if c < min_c: min_c = c; bw = w; bidx = i
        return min_c, bw, bidx

    def _generate_bezier_wps(self, p0, yaw0, p3, yaw3, num_points=30):
        """生成从当前位置到目标位置的平滑贝塞尔引导曲线"""
        dist = np.linalg.norm(p3 - p0)
        angle_to_target = np.arctan2(p3[1] - p0[1], p3[0] - p0[0])
        diff_start = np.abs((yaw0 - angle_to_target + np.pi) % (2*np.pi) - np.pi)
        diff_end = np.abs((yaw3 - angle_to_target + np.pi) % (2*np.pi) - np.pi)
        L1 = np.clip(dist * (0.3 + 0.2 * (diff_start / np.pi)), 0.1, 2.5)
        L2 = np.clip(dist * (0.3 + 0.2 * (diff_end / np.pi)), 0.1, 2.5)
        p1 = p0 + L1 * np.array([np.cos(yaw0), np.sin(yaw0)])
        p2 = p3 - L2 * np.array([np.cos(yaw3), np.sin(yaw3)])
        wps = []
        for t in np.linspace(0, 1.0, num_points):
            t_inv = 1.0 - t
            pos = t_inv**3 * p0 + 3 * t_inv**2 * t * p1 + 3 * t_inv * t**2 * p2 + t**3 * p3
            dp = 3 * t_inv**2 * (p1 - p0) + 6 * t_inv * t * (p2 - p1) + 3 * t**2 * (p3 - p2)
            tangent_yaw = np.arctan2(dp[1], dp[0])
            wps.append({'pos': pos, 'yaw': tangent_yaw})
        return wps

    def _is_bezier_safe(self, idx, wps):
        """检查整条贝塞尔曲线是否安全（不出界、不撞墙、不撞其他车）"""
        for wp in wps[::2]:
            corners = self.get_rect_corners(wp['pos'], wp['yaw'], self.agent_length + 0.1, self.agent_width + 0.1)
            if not self._is_rect_completely_inside(corners, self.boundary_coords): return False
            if self._is_rect_colliding_with_segments(corners, self.obs_segments): return False
            for j in range(self.max_agents):
                if idx != j and self.active_mask[j]:
                    if np.linalg.norm(wp['pos'] - self.positions[j]) < 1.0: return False
        return True

    def _compute_potential(self, idx):
        pos = self.positions[idx]; yaw = self.yaws[idx]
        if self.phases[idx] == "exit":
            target = {'pos': self.exit_pos, 'yaw': self.exit_yaw}
        else:
            target = self.locked_w_poses[idx]
            
        d = np.linalg.norm(pos - target['pos'])
        diff = (yaw - target['yaw'] + np.pi) % (2 * np.pi) - np.pi
        y_err = np.abs(diff)
        dx, dy = pos[0] - target['pos'][0], pos[1] - target['pos'][1]
        cte = np.abs(dx * np.sin(target['yaw']) - dy * np.cos(target['yaw']))
        
        # 新增：航向对齐项 - 引导智能体朝向目标点
        heading_to_target = np.arctan2(target['pos'][1] - pos[1], target['pos'][0] - pos[0])
        heading_align = np.abs((heading_to_target - yaw + np.pi) % (2 * np.pi) - np.pi)
        
        return 1.0 * d + 5.0 * y_err + 3.0 * cte + 2.0 * heading_align

    def step(self, actions):
        rewards = np.zeros(self.max_agents, dtype=np.float32)
        dones = np.zeros(self.max_agents, dtype=bool)
        infos = [{} for _ in range(self.max_agents)]
        
        self.current_step += 1
        self.steps_since_last_spawn += 1
        
        # 1. 尝试产生新车
        if self.steps_since_last_spawn >= self.spawn_interval:
            self._try_spawn()

        # 2. 预测并更新所有车辆位置 (第一阶段：动作执行)
        for i in range(self.max_agents):
            if not self.active_mask[i]:
                rewards[i] = 0.0
                continue
            
            action = actions[i]
            
            # --- Safety Shield (A3) ---
            # 如果动作不安全，强制覆盖为刹车
            if not self._is_action_safe(i, action):
                action = np.array([0.0, 0.0], dtype=np.float32) # 紧急刹车
                infos[i]['safety_intervention'] = True
                rewards[i] -= 1.0 # 给一个小惩罚，告诉它"你刚才想做危险动作"
            
            v, delta = np.clip(action[0], 0.0, 0.5), np.clip(action[1], -self.max_steer, self.max_steer)
            
            if self.goal_switched[i] and self.phases[i] == "forward_rl":
                v = np.clip(v, 0.0, 0.2)
            
            # --- 物理位姿更新 ---
            if self.phases[i] == "forward_rl" or self.phases[i] == "exit":
                if np.abs(delta) > 1e-6:
                    tr = self.wheelbase / np.tan(delta)
                    aw = v / tr
                    new_yaw = (self.yaws[i] + aw + np.pi) % (2*np.pi) - np.pi
                    self.positions[i][0] += tr * (np.sin(new_yaw) - np.sin(self.yaws[i]))
                    self.positions[i][1] += tr * (np.cos(self.yaws[i]) - np.cos(new_yaw))
                    self.yaws[i] = new_yaw
                else:
                    self.positions[i][0] += v * np.cos(self.yaws[i])
                    self.positions[i][1] += v * np.sin(self.yaws[i])
                
                # 目标与阶段逻辑 (Stage 1)
                if self.phases[i] == "forward_rl":
                    dist_to_site = np.linalg.norm(self.positions[i] - self.target_sites[i]['pos'])
                    if not self.goal_switched[i] and dist_to_site <= self.decision_dist:
                        _, g_w, _ = self._calculate_best_w(i)
                        self.locked_w_poses[i] = g_w
                        self.goal_switched[i] = True
                        self.goal_switch_step[i] = self.current_step  # 记录切换时间戳
                    
                    locked_pos = self.locked_w_poses[i]['pos']
                    dist_err = np.linalg.norm(self.positions[i] - locked_pos)
                    yaw_err = (self.yaws[i] - self.locked_w_poses[i]['yaw'] + np.pi) % (2*np.pi) - np.pi
                    
                    dx, dy = self.positions[i][0] - locked_pos[0], self.positions[i][1] - locked_pos[1]
                    target_yaw = self.locked_w_poses[i]['yaw']
                    err_long = dx * np.cos(target_yaw) + dy * np.sin(target_yaw)
                    err_lat = -dx * np.sin(target_yaw) + dy * np.cos(target_yaw)
                    
                    # === 主路径: 严格几何框判定 → 标准圆弧倒车 ===
                    if np.abs(err_long) < 0.4 and np.abs(err_lat) < 0.25 and np.abs(yaw_err) < 0.5 and v < 0.1:
                        self.low_speed_frames[i] += 1
                        if self.low_speed_frames[i] >= 3:
                            self.phases[i] = "reversing"; self.arc_progress[i] = 0.0
                            self.near_w_steps[i] = 0
                            self.reversing_stuck_steps[i] = 0
                            self.positions[i] = locked_pos.copy(); self.yaws[i] = target_yaw
                            rewards[i] += 100.0
                            self.bezier_wps[i] = None  # 使用标准圆弧
                    # === 备用路径 (仅超时将至时): 动态贝塞尔曲线倒车 ===
                    # near_w_steps > 25 = 已在 W 点附近徘徊约 2.5 秒，说明精准停车失败，启用容错
                    elif dist_err < 1.5 and np.abs(yaw_err) < 1.0 and v < 0.1 and self.near_w_steps[i] > 25:
                        self.low_speed_frames[i] += 1
                        if self.low_speed_frames[i] >= 3:
                            target_site = self.target_sites[i]
                            start_pos = self.positions[i].copy()
                            start_yaw = self.yaws[i]
                            entry_yaw = target_site['yaw']
                            sdx, sdy = start_pos[0] - target_site['pos'][0], start_pos[1] - target_site['pos'][1]
                            g_to_v_yaw = np.arctan2(sdy, sdx)
                            if np.cos(entry_yaw) * np.cos(g_to_v_yaw) + np.sin(entry_yaw) * np.sin(g_to_v_yaw) > 0:
                                entry_yaw = (entry_yaw + np.pi) % (2*np.pi)
                            test_wps = self._generate_bezier_wps(start_pos, start_yaw, target_site['pos'], entry_yaw)
                            if self._is_bezier_safe(i, test_wps):
                                self.phases[i] = "reversing"; self.arc_progress[i] = 0.0
                                self.near_w_steps[i] = 0; self.reversing_stuck_steps[i] = 0
                                self.bezier_wps[i] = test_wps
                                rewards[i] += 60.0  # 容错路径打折（满分 100 → 60）
                    else:
                        self.low_speed_frames[i] = 0
                        
                    # 失败监测 (错失/超时)
                    if self.goal_switched[i]:
                        if dist_err < 1.5: self.near_w_steps[i] += 1
                        else: self.near_w_steps[i] = 0 # Reset if not near W
                        
                        if not self.approach_started[i]:
                            self.approach_started[i] = True
                            self.min_dist_to_w[i] = dist_err
                        else:
                            if dist_err > self.min_dist_to_w[i]: self.overshoot_steps[i] += 1
                            else: self.min_dist_to_w[i] = dist_err; self.overshoot_steps[i] = 0
                        
                        if self.near_w_steps[i] > 200 or (self.overshoot_steps[i] > 10 and dist_err > self.min_dist_to_w[i] + 1.0):
                            # === 诊断日志：分析 W-Point Miss 的真实原因 ===
                            nearby = []
                            for j in range(self.max_agents):
                                if j != i and self.active_mask[j]:
                                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                                    if d < 3.0:
                                        nearby.append(f"Veh#{j}({self.phases[j]}):{d:.2f}m")
                            cause = "near_w>100" if self.near_w_steps[i] > 100 else "overshoot"
                            print(f"  [W-MISS诊断] Veh#{i} | 原因:{cause} | dist_err:{dist_err:.2f} | near_w:{self.near_w_steps[i]} | "
                                  f"overshoot:{self.overshoot_steps[i]} | 3m内车辆:{nearby if nearby else '无'}")
                            rewards[i] -= 50.0
                            dones[i] = True; infos[i]['timeout'] = True; infos[i]['timeout_reason'] = 'w_point_miss'
                
                # 完成离场 (Stage 3)
                elif self.phases[i] == "exit":
                    if np.linalg.norm(self.positions[i] - self.exit_pos) < 0.4:
                        rewards[i] += 500.0; dones[i] = True; infos[i]['success'] = True
                        
            elif self.phases[i] == "reversing":
                # 倒车逻辑 (Radar Pause)
                next_progress = self.arc_progress[i] + 0.02
                temp_pos, temp_yaw = self._get_pose_on_arc(i, next_progress)
                temp_corners = self.get_rect_corners(temp_pos, temp_yaw, self.agent_length, self.agent_width)
                
                blocked = False
                for j in range(self.max_agents):
                    if i != j and self.active_mask[j]:
                        cj = self.get_rect_corners(self.positions[j], self.yaws[j], self.agent_length, self.agent_width)
                        if self._rectangles_intersect(temp_corners, cj):
                            blocked = True; break
                
                if not blocked:
                    self.arc_progress[i] = next_progress; self.positions[i], self.yaws[i] = temp_pos, temp_yaw
                    self.reversing_stuck_steps[i] = 0
                    if self.arc_progress[i] >= 1.0:
                        self.phases[i] = "exit"; rewards[i] += 100.0
                        self.positions[i] = self.target_sites[i]['pos'].copy(); self.yaws[i] = self.target_sites[i]['yaw']
                else:
                    rewards[i] -= 0.1; self.reversing_stuck_steps[i] += 1
                    if self.reversing_stuck_steps[i] > 150:
                        rewards[i] -= 50.0
                        dones[i] = True; infos[i]['timeout'] = True; infos[i]['timeout_reason'] = 'reversing_stuck'

        # 3. 统一碰撞判定 (第二阶段：状态结算)
        collision_indices = set()
        for i in range(self.max_agents):
            if not self.active_mask[i] or dones[i]: continue
            
            ci = self.get_rect_corners(self.positions[i], self.yaws[i], self.agent_length, self.agent_width)
            # 环境碰撞
            if not self._is_rect_completely_inside(ci, self.boundary_coords) or self._is_rect_colliding_with_segments(ci, self.obs_segments):
                collision_indices.add(i); infos[i]['collision_type'] = 'static'
                continue
                
            # 三车互撞检测
            for j in range(i + 1, self.max_agents):
                if self.active_mask[j] and not dones[j]:
                    cj = self.get_rect_corners(self.positions[j], self.yaws[j], self.agent_length, self.agent_width)
                    if self._rectangles_intersect(ci, cj):
                        collision_indices.add(i); collision_indices.add(j)
                        infos[i]['collision_type'] = 'inter_vehicle'
                        infos[j]['collision_type'] = 'inter_vehicle'

        # 4. 执行销毁与奖励结算
        for i in range(self.max_agents):
            if not self.active_mask[i]: continue
            
            if i in collision_indices:
                # Apply penalty based on collision type
                penalty = -100.0 if infos[i].get('collision_type') == 'static' else -50.0
                rewards[i] += penalty
                dones[i] = True; infos[i]['collision'] = True
            
            if dones[i]:
                self._despawn(i)
                continue
            
            # 常规奖励计算
            if self.phases[i] != "reversing":
                lidar_raw = self._get_lidar_readings(i)
                min_l = np.min(lidar_raw)
                if min_l < 0.4: rewards[i] -= 0.1 * (1.0 - min_l / 0.4)
                
                curr_cost = self._compute_potential(i)
                rewards[i] += (self.prev_potential_costs[i] - curr_cost) * 10.0
                rewards[i] -= 0.05 # 时间惩罚
                
                # 低速惩罚 (但在 W 点附近不惩罚，鼓励减速)
                if actions[i][0] < 0.05 and not self.goal_switched[i]:
                    rewards[i] -= 0.05

                # APF 车间斥力 (优先级感知，不对称惩罚)
                my_pri = self._get_priority(i)
                for j in range(self.max_agents):
                    if i != j and self.active_mask[j]:
                        dist_ij = np.linalg.norm(self.positions[i] - self.positions[j])
                        if dist_ij < 2.0:
                            other_pri = self._get_priority(j)
                            # 对方优先级更高 → 我该让路 → 惩罚 3 倍
                            multiplier = 3.0 if other_pri > my_pri else 1.0
                            rewards[i] -= multiplier * 0.05 * (2.0 - dist_ij)
                            if dist_ij < 1.0:
                                rewards[i] -= multiplier * 0.1 / (dist_ij + 0.1)**2
                
                # 动作平滑与速度激励
                rewards[i] -= 0.1 * np.abs(actions[i][1] - self.prev_actions[i][1])
                rewards[i] -= 0.1 * np.abs(actions[i][0] - self.prev_actions[i][0])
                if not self.goal_switched[i]: rewards[i] += 0.01 * actions[i][0]
                
                self.prev_potential_costs[i] = curr_cost
            
            self.prev_actions[i] = actions[i].copy()
            self.traj_histories[i].append(self.positions[i].copy())

        # --- 全局超时逻辑 (硬截止) ---
        # 如果整场时间到了，强制所有槽位 done，触发 VecEnv 层次的全局 reset
        if self.current_step >= self.max_steps:
            for i in range(self.max_agents):
                if self.active_mask[i] and not dones[i]:
                    infos[i]['timeout'] = True; infos[i]['timeout_reason'] = 'global_limit'
            dones.fill(True)

        return self._get_all_obs_dict(), rewards, dones, dones, infos

    def _move_along_arc(self, idx, progress):
        pos, yaw = self._get_pose_on_arc(idx, progress)
        self.positions[idx] = pos
        self.yaws[idx] = yaw

    def _is_action_safe(self, idx, action):
        """
        Action Masking / Safety Shield (A3):
        预测执行该动作后下一帧的位置，如果会撞车或撞墙，返回 False。
        """
        # 1. 简单的运动学预测
        v, delta = np.clip(action[0], 0.0, 0.5), np.clip(action[1], -self.max_steer, self.max_steer)
        
        # 复制当前状态进行模拟
        curr_pos = self.positions[idx].copy()
        curr_yaw = self.yaws[idx]
        
        if np.abs(delta) > 1e-6:
            tr = self.wheelbase / np.tan(delta)
            aw = v / tr
            new_yaw = (curr_yaw + aw + np.pi) % (2*np.pi) - np.pi
            pred_pos = curr_pos + np.array([tr * (np.sin(new_yaw) - np.sin(curr_yaw)),
                                            tr * (np.cos(curr_yaw) - np.cos(new_yaw))])
        else:
            new_yaw = curr_yaw
            pred_pos = curr_pos + np.array([v * np.cos(curr_yaw), v * np.sin(curr_yaw)])
            
        # 2. 碰撞检测
        # 获取预测位置的矩形角点
        pred_corners = self.get_rect_corners(pred_pos, new_yaw, self.agent_length, self.agent_width)
        
        # 检查静态环境碰撞
        if not self._is_rect_completely_inside(pred_corners, self.boundary_coords) or \
           self._is_rect_colliding_with_segments(pred_corners, self.obs_segments):
            return False
            
        # 检查与其他车辆碰撞
        for j in range(self.max_agents):
            if idx == j or not self.active_mask[j]: continue
            
            # 这里简单起见，假设其他车辆位置不变（或者也可以假设它们保持当前速度）
            # 为了保守起见（Safety Shield），假设它们静止是最低限度的检查，
            # 更好的做法是假设它们也执行上一步的动作，但这里为了计算效率暂且用静态位置。
            other_corners = self.get_rect_corners(self.positions[j], self.yaws[j], self.agent_length, self.agent_width)
            if self._rectangles_intersect(pred_corners, other_corners):
                return False
                
        return True

    def _get_pose_on_arc(self, idx, progress):
        progress = np.clip(progress, 0.0, 1.0)
        
        # 动态贝塞尔曲线路径 (备用路径生成的)
        if self.bezier_wps[idx] is not None:
            wps = self.bezier_wps[idx]
            N = len(wps) - 1
            f_idx = progress * N
            i1 = int(np.floor(f_idx))
            i2 = min(i1 + 1, N)
            alpha = f_idx - i1
            p1, p2 = wps[i1]['pos'], wps[i2]['pos']
            pos = p1 + alpha * (p2 - p1)
            # 倒车：车头朝运动轨迹的反方向 (翻转 180°)
            if np.linalg.norm(p2 - p1) > 1e-4:
                tangent_yaw = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                yaw = (tangent_yaw + np.pi) % (2*np.pi) - np.pi
            else:
                yaw = wps[i1]['yaw']
            return pos, yaw
        
        # 标准圆弧路径 (主路径)
        w_pose = self.locked_w_poses[idx]
        L = w_pose['L']; dt = w_pose['delta_theta']; r = w_pose['r']
        pos_d = self.target_sites[idx]['pos']; yaw_d = self.target_sites[idx]['yaw']
        arc_t = (1.0 - progress) * dt
        if np.abs(dt) < 1e-6:
            dist_along = (1.0 - progress) * L
            pos = pos_d + np.array([np.cos(yaw_d), np.sin(yaw_d)]) * dist_along
            yaw = yaw_d
        else:
            rs = np.sign(dt)
            xl = r * np.sin(np.abs(arc_t))
            yl = rs * r * (1.0 - np.cos(np.abs(arc_t)))
            c, s = np.cos(yaw_d), np.sin(yaw_d)
            pos = pos_d + np.array([c * xl - s * yl, s * xl + c * yl])
            yaw = yaw_d + arc_t
        return pos, yaw

    def _despawn(self, idx):
        self.active_mask[idx] = False
        self.phases[idx] = "inactive"
        self.target_sites[idx] = None
        
        # 立即尝试重生新车（如果入口空闲）
        self._try_spawn()

    def _rectangles_intersect(self, corners1, corners2):
        # 简单的 SAT 碰撞检测
        def get_axes(c):
            axes = []
            for i in range(4):
                p1, p2 = c[i], c[(i+1)%4]
                edge = p2 - p1
                axes.append(np.array([-edge[1], edge[0]])) # 法向量
            return axes
        
        axes = get_axes(corners1) + get_axes(corners2)
        for axis in axes:
            axis = axis / np.linalg.norm(axis)
            proj1 = [np.dot(p, axis) for p in corners1]
            proj2 = [np.dot(p, axis) for p in corners2]
            if min(proj1) > max(proj2) or min(proj2) > max(proj1):
                return False
        return True

    def render_frame(self):
        """返回当前环境状态的 RGB 图像数组 (使用 OO API 确保稳健性)"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        import numpy as np
        
        # 显式创建，不使用缓存
        fig = Figure(figsize=(8, 8), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # 1. 绘制背景色 (随步数闪烁，强制肉眼识别变化)
        bg_color = '#ffffff' if (self.current_step // 2) % 2 == 0 else '#f0f0f0'
        fig.patch.set_facecolor(bg_color)
        
        # 2. 绘图逻辑
        self._draw_state(ax)
        
        # 强制渲染
        canvas.draw()
        
        rgba = np.asarray(canvas.buffer_rgba())
        from PIL import Image
        return np.array(Image.fromarray(rgba).convert('RGB'))

    def save_training_frame(self, save_path=None):
        """保存当前环境状态到文件"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 10))
        self._draw_state(ax)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def _draw_state(self, ax):
        # 1. 绘制障碍物
        for seg in self.obs_segments:
            ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color='blue', linewidth=2, alpha=0.5)
        
        dl, dw = 0.7, 0.4
        # 2. 绘制所有排土位 (D点)
        for site in self.dumping_sites:
            pos, yaw = site['pos'], site['yaw']
            is_any_target = False
            locked_w = None
            for i in range(self.max_agents):
                if self.active_mask[i] and self.target_sites[i] is not None:
                    if np.array_equal(pos, self.target_sites[i]['pos']):
                        is_any_target = True
                        locked_w = self.locked_w_poses[i]
                        break
            
            sc, ss = np.cos(yaw), np.sin(yaw)
            rot = np.array([[sc, -ss], [ss, sc]])
            corners_rect = np.array([[dl/2, dw/2], [dl/2, -dw/2], [-dl/2, -dw/2], [-dl/2, dw/2], [dl/2, dw/2]])
            rotated_corners = corners_rect @ rot.T + pos
            
            color = 'orange' if is_any_target else 'green'
            alpha_s = 0.9 if is_any_target else 0.3
            lw_s = 2.5 if is_any_target else 1.0
            
            ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], color=color, linewidth=lw_s, alpha=alpha_s)
            ax.fill(rotated_corners[:, 0], rotated_corners[:, 1], color=color, alpha=0.1)
            ax.arrow(pos[0], pos[1], sc * 0.2, ss * 0.2, head_width=0.03, head_length=0.06, fc=color, ec=color, alpha=alpha_s)
            
            if is_any_target:
                ax.plot(pos[0], pos[1], marker='*', markersize=15, color='gold', markeredgecolor='orange', zorder=15)
                if locked_w:
                    w_pos, w_yaw = locked_w['pos'], locked_w['yaw']
                    wc, ws = np.cos(w_yaw), np.sin(w_yaw)
                    w_rot = np.array([[wc, -ws], [ws, wc]])
                    r_w = corners_rect @ w_rot.T + w_pos
                    ax.plot(r_w[:, 0], r_w[:, 1], color='red', linewidth=2.5, alpha=1.0)
                    
                    # 绘制倒车引导路径
                    dt_v, rv = locked_w['delta_theta'], locked_w['r']
                    if np.abs(dt_v) < 1e-6:
                        ax.plot([pos[0], w_pos[0]], [pos[1], w_pos[1]], 'k--', linewidth=0.5, alpha=0.3)
                    else:
                        ts = np.linspace(0, dt_v, 15); rs_val = np.sign(dt_v)
                        xa, ya = rv*np.sin(np.abs(ts)), rs_val*rv*(1.0-np.cos(np.abs(ts)))
                        xg = pos[0]+sc*xa-ss*ya; yg = pos[1]+ss*xa+sc*ya
                        ax.plot(xg, yg, color='black', linestyle='--', linewidth=1.0, alpha=0.5)

        # 3. 绘制智能体
        colors = ['purple', 'maroon', 'indigo']
        for i in range(self.max_agents):
            if not self.active_mask[i]: continue
            
            a_pos, a_yaw = self.positions[i], self.yaws[i]
            ca, sa = np.cos(a_yaw), np.sin(a_yaw)
            rot_aa = np.array([[ca, -sa], [sa, ca]])
            corners_a = np.array([[self.agent_length/2, self.agent_width/2], [self.agent_length/2, -self.agent_width/2], [-self.agent_length/2, -self.agent_width/2], [-self.agent_length/2, self.agent_width/2], [self.agent_length/2, self.agent_width/2]])
            r_c_aa = corners_a @ rot_aa.T + a_pos
            color = colors[i % len(colors)]
            ax.plot(r_c_aa[:, 0], r_c_aa[:, 1], color=color, linewidth=2)
            ax.arrow(a_pos[0], a_pos[1], ca*0.4, sa*0.4, head_width=0.08, head_length=0.12, fc=color, ec=color)
            ax.text(a_pos[0], a_pos[1]+0.3, f"Vehicle#{i}", color=color, fontsize=9, fontweight='bold')
            
            # 轨迹
            traj = np.array(self.traj_histories[i])
            if len(traj) > 1:
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, alpha=0.6)

        # 4. 绘制出口点
        ax.plot(self.exit_pos[0], self.exit_pos[1], marker='s', markersize=12, color='cyan', markeredgecolor='blue', zorder=15)
        
        ax.set_xlim(-1, 11); ax.set_ylim(-1, 11); ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # 5. 绘制极其显眼的时间戳 (左上角)
        # 增加一个 "Heartbeat" (. 或 _) 随步数变化，证明视频在动
        heartbeat = "●" if (self.current_step // 5) % 2 == 0 else "○"
        info_text = f"STEP: {self.current_step:04d} {heartbeat}\nTIME: {self.current_step*0.1:.1f}s\nACTIVE: {np.sum(self.active_mask)}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, va='top', ha='left', 
                fontsize=16, fontweight='bold', family='monospace',
                bbox=dict(facecolor='lime' if (self.current_step // 10) % 2 == 0 else 'yellow', 
                          alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))
        
        ax.set_title(f"Multi-Agent Evaluation Run", fontsize=16, weight='bold', pad=25)

    def _is_rect_completely_inside(self, corners, polygon):
        for pt in corners:
            if not self._is_point_in_polygon(pt, polygon): return False
        return True

    def _is_point_in_polygon(self, point, polygon):
        x, y = point; n = len(polygon); inside = False; p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y: xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                if p1x == p2x or x <= xinters: inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _is_rect_colliding_with_segments(self, corners, segments):
        rect_edges = [[corners[i], corners[(i+1)%4]] for i in range(4)]
        for edge in rect_edges:
            for seg in segments:
                if self._line_segments_intersect(edge[0], edge[1], seg[0], seg[1]): return True
        return False

    @property
    def has_active_agents(self):
        return np.any(self.active_mask)

    @property
    def has_running_agents(self):
        # 只要有一个活跃智能体，且它的轨迹长度 > 15 (说明跑了15步以上)
        for i in range(self.max_agents):
            if self.active_mask[i] and len(self.traj_histories[i]) > 15:
                # 还可以加一个 check: 距离起点有一段距离 (避免原地不动的车被算进去)
                # if np.linalg.norm(self.positions[i] - self.entry_pos) > 1.0:
                return True
        return False

    def _line_segments_intersect(self, p1, p2, p3, p4):
        def ccw(A, B, C): return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

class MultiAgentVecEnv(VecEnv):
    """单进程多智能体包装器"""
    def __init__(self, env: MultiVehicleLiDAR2DEnv):
        self.env = env
        super().__init__(env.max_agents, env.observation_space, env.action_space)
        self.actions = None

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs_dict, rews, terms, truncs, infos = self.env.step(self.actions)
        dones = np.logical_or(terms, truncs)
        if np.all(dones):
            new_obs_dict, _ = self.env.reset()
            obs_dict = new_obs_dict
        return obs_dict, rews, dones, infos

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self.env, method_name)
        return [method(*method_args, **method_kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

# --- 以下为新增的多进程加速组件 ---

import multiprocessing as mp
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper



def _multi_agent_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    max_agents = env.max_agents
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, rews, terms, truncs, infos = env.step(data)
                dones = np.logical_or(terms, truncs)
                if np.all(dones):
                    obs, _ = env.reset()
                remote.send((obs, rews, dones, infos))
            elif cmd == "reset":
                obs, _ = env.reset()
                remote.send(obs)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "seed":
                remote.send(env.reset(seed=data))
            else:
                raise NotImplementedError(f"Unknown command {cmd}")
    except EOFError:
        pass


class SubprocMultiAgentVecEnv(VecEnv):
    """
    多进程多智能体 VecEnv。
    n_procs: 进程数 (例如 12)
    max_agents: 每个环境中的智能体数 (例如 3)
    总 envs = n_procs * max_agents
    """
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        n_procs = len(env_fns)
        
        # 临时创建一个环境获取参数
        temp_env = env_fns[0]()
        self.max_agents_per_env = temp_env.max_agents
        num_envs = n_procs * self.max_agents_per_env
        super().__init__(num_envs, temp_env.observation_space, temp_env.action_space)
        temp_env.close()

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_procs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # 设置 daemon=True 确保主进程退出时子进程也退出
            process = mp.Process(target=_multi_agent_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def step_async(self, actions):
        # actions shape: (num_envs, action_dim) -> (n_procs * 3, 2)
        # 需要将其拆分为 12 个 (3, 2) 给各个进程
        for i, remote in enumerate(self.remotes):
            start = i * self.max_agents_per_env
            end = start + self.max_agents_per_env
            remote.send(("step", actions[start:end]))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        # results: list of (obs_dict, rews, dones, infos)
        # obs_dict: {"obs": (max_agents, self_obs_dim), "state": (max_agents, state_dim)}
        
        all_obs = np.concatenate([r[0]["obs"] for r in results])
        all_states = np.concatenate([r[0]["state"] for r in results])
        all_rews = np.concatenate([r[1] for r in results])
        all_dones = np.concatenate([r[2] for r in results])
        all_infos = []
        for r in results:
            all_infos.extend(r[3])
            
        combined_obs = {"obs": all_obs, "state": all_states}
        return combined_obs, all_rews, all_dones, all_infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        # results: list of obs_dict
        all_obs = np.concatenate([r["obs"] for r in results])
        all_states = np.concatenate([r["state"] for r in results])
        return {"obs": all_obs, "state": all_states}

    def close(self):
        if self.closed: return
        if self.waiting:
            for remote in self.remotes: remote.recv()
        for remote in self.remotes: remote.send(("close", None))
        for process in self.processes: process.join()
        self.closed = True

    def get_attr(self, attr_name, indices=None):
        target_indices = self._get_target_indices(indices)
        target_procs = sorted(list(set([i // self.max_agents_per_env for i in target_indices])))
        
        results = {}
        for proc_idx in target_procs:
            self.remotes[proc_idx].send(("get_attr", attr_name))
        
        for proc_idx in target_procs:
            results[proc_idx] = self.remotes[proc_idx].recv()
            
        final_results = []
        for i in target_indices:
            proc_idx = i // self.max_agents_per_env
            final_results.append(results[proc_idx])
        return final_results

    def set_attr(self, attr_name, value, indices=None):
        target_indices = self._get_target_indices(indices)
        target_procs = sorted(list(set([i // self.max_agents_per_env for i in target_indices])))
        
        results = {}
        for proc_idx in target_procs:
            self.remotes[proc_idx].send(("set_attr", (attr_name, value)))
            
        for proc_idx in target_procs:
            results[proc_idx] = self.remotes[proc_idx].recv()
            
        final_results = []
        for i in target_indices:
            proc_idx = i // self.max_agents_per_env
            final_results.append(results[proc_idx])
        return final_results

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        target_indices = self._get_target_indices(indices)
        target_procs = sorted(list(set([i // self.max_agents_per_env for i in target_indices])))
        
        results = {}
        for proc_idx in target_procs:
            self.remotes[proc_idx].send(("env_method", (method_name, method_args, method_kwargs)))
            
        for proc_idx in target_procs:
            results[proc_idx] = self.remotes[proc_idx].recv()
            
        final_results = []
        for i in target_indices:
            proc_idx = i // self.max_agents_per_env
            final_results.append(results[proc_idx])
        return final_results

    def env_is_wrapped(self, wrapper_class, indices=None):
        target_indices = self._get_target_indices(indices)
        # For now, just return False as we don't support dynamic wrappers in subprocesses easily yet
        # or properly implement it if needed, but the original code had [False] * num_envs.
        # Let's match the interface `return [bool]`.
        return [False] * len(target_indices)

    def seed(self, seed=None):
        target_indices = self._get_target_indices(None)
        # Seed logic might be slightly different: we usually want to seed all environments
        # Each process has its own random state, so we send seed to all
        for remote in self.remotes: 
            remote.send(("seed", seed))
        
        # Expect 1 response per process
        proc_results = [remote.recv() for remote in self.remotes]
        
        # Return list of results for each agent (duplicated per process)
        final_results = []
        for i in target_indices:
            proc_idx = i // self.max_agents_per_env
            final_results.append(proc_results[proc_idx])
        return final_results

    def _get_target_indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        return indices
