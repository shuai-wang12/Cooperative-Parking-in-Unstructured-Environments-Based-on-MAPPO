import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import os

class CustomLiDAR2DEnv(gym.Env):
    """
    一个包含线段障碍物的二维导航环境，使用位姿代价函数引导智能体。
    引入“距离触发”目标更新逻辑：在距离站点中心 1.5m 时执行一次性目标优化重选并锁定。
    """
    def __init__(self):
        super(CustomLiDAR2DEnv, self).__init__()

        # 障碍物线段
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

        self.max_steps = 350
        self.n_lidar_rays = 24
        self.lidar_range = 5.0
        self.agent_length = 0.5
        self.agent_width = 0.2
        self.wheelbase = 0.3  
        self.min_turning_radius = 1.25 
        self.max_steer = np.arctan(self.wheelbase / self.min_turning_radius) 
        
        self.decision_dist = 4.0 # 增加决策距离，给智能体更充裕的时间提前对准换向点
        self.obs_segments = np.array(self.obstacle_lines)
        self.dumping_sites = self.sample_dumping_sites()
        
        self.action_space = spaces.Box(low=np.array([0.0, -self.max_steer]), high=np.array([0.5, self.max_steer]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24 + 2 + 2 + 2,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if hasattr(self, 'traj_history'):
            self.last_episode_data = {
                'traj': self.traj_history.copy(), 'target_site': self.target_site,
                'locked_w_pose': self.locked_w_pose, 'agent_pos': self.agent_pos.copy(),
                'agent_yaw': self.agent_yaw, 'final_step': self.current_step,
                'goal_switched': getattr(self, 'goal_switched', False),
                'phase': getattr(self, 'phase', 'forward_rl'),
                'exit_pos': getattr(self, 'exit_pos', np.array([0, 0])).copy()
            }
        else: self.last_episode_data = None

        # 入口点微扰动：位置 ±0.2m，角度 ±5°（约 0.087 rad）
        base_pos = np.array([2.08, 8.0], dtype=np.float32)
        base_yaw = -1.901295
        pos_noise = self.np_random.uniform(-0.1, 0.1, size=2).astype(np.float32)  # ±10cm 扰动
        yaw_noise = self.np_random.uniform(-0.087, 0.087)  # ±5度
        self.agent_pos = base_pos + pos_noise
        self.agent_yaw = base_yaw + yaw_noise
        self.traj_history = [self.agent_pos.copy()]
        self.target_site = self.np_random.choice(self.dumping_sites)
        
        # 初始选择最优点锁定
        _, bw, bidx = self._calculate_global_best_w()
        self.locked_w_pose = bw
        self.locked_idx = bidx
        self.goal_switched = False
        
        # Stage 2/3 状态变量
        self.phase = "forward_rl"  # forward_rl / reversing / exit
        self.arc_progress = 0.0    # 倒车曲线进度 [0, 1]
        self.exit_progress = 0.0   # 出库进度 [0, 1]
        
        # 出口点：固定坐标
        self.exit_pos = np.array([3.58, 8.0], dtype=np.float32)
        self.exit_yaw = base_yaw + np.pi  # 出口时朝向与入口相反（逆转180度）
        
        # 使用统一势能函数初始化 prev_min_cost
        self.prev_min_cost = self._compute_potential(self.agent_pos, self.agent_yaw, self.locked_w_pose)
        self.current_step = 0
        self.prev_action = np.zeros(2)
        self.has_reached_pos = False
        
        # "错过即失败" 追踪变量
        self.min_dist_to_w = float('inf')  # 记录到换向点的最小距离
        self.approach_started = False       # 是否已进入接近阶段
        self.overshoot_steps = 0            # 连续超调步数
        
        return self._get_obs(), {}

    def _compute_potential(self, pos, yaw, target_pose):
        """统一的势能函数计算，包含侧向偏差惩罚"""
        d = np.linalg.norm(pos - target_pose['pos'])
        # 正确的角度差计算顺序：先算差值，再归一化，最后取绝对值
        diff = yaw - target_pose['yaw']
        diff = (diff + np.pi) % (2 * np.pi) - np.pi  # 归一化到 [-pi, pi]
        y_err = np.abs(diff)
        
        # 侧向偏差 (Cross Track Error): 车辆到目标朝向直线的垂直距离
        # 目标朝向单位向量
        target_yaw = target_pose['yaw']
        dx = pos[0] - target_pose['pos'][0]
        dy = pos[1] - target_pose['pos'][1]
        # CTE = 叉积的绝对值 = |dx * sin(target_yaw) - dy * cos(target_yaw)|
        cte = np.abs(dx * np.sin(target_yaw) - dy * np.cos(target_yaw))
        
        # 势能 = 距离 + 角度误差(大幅提高权重) + 侧向偏差(提高权重)
        return 1.0 * d + 6.0 * y_err + 3.0 * cte

    def _calculate_global_best_w(self):
        """选择最优换向点，引入进场对齐度惩罚（老司机直觉版）"""
        min_c = float('inf')
        bw = None
        bidx = -1
        
        # 第一轮：只考虑前方 ±90° 扇区内的点
        for i, w in enumerate(self.target_site['w_poses']):
            d = np.linalg.norm(self.agent_pos - w['pos'])
            dx = w['pos'][0] - self.agent_pos[0]
            dy = w['pos'][1] - self.agent_pos[1]
            heading_to_point = np.arctan2(dy, dx)
            
            h_diff = heading_to_point - self.agent_yaw
            h_diff = (h_diff + np.pi) % (2 * np.pi) - np.pi
            
            # 硬性过滤：跳过身后的点
            if np.abs(h_diff) > np.pi / 2:
                continue
            
            entry_align = heading_to_point - w['yaw']
            entry_align = (entry_align + np.pi) % (2 * np.pi) - np.pi
            
            # 大幅提高对齐权重，降低距离权重，确保选出的点是真正“顺路”且“易入”的
            c = 0.2 * d + 3.0 * np.abs(h_diff) + 6.0 * np.abs(entry_align)
            
            if c < min_c:
                min_c = c
                bw = w
                bidx = i
        
        # 【后备逻辑】如果所有点都被过滤掉了，回退到无过滤模式
        if bw is None:
            for i, w in enumerate(self.target_site['w_poses']):
                d = np.linalg.norm(self.agent_pos - w['pos'])
                dx = w['pos'][0] - self.agent_pos[0]
                dy = w['pos'][1] - self.agent_pos[1]
                heading_to_point = np.arctan2(dy, dx)
                
                h_diff = heading_to_point - self.agent_yaw
                h_diff = (h_diff + np.pi) % (2 * np.pi) - np.pi
                
                entry_align = heading_to_point - w['yaw']
                entry_align = (entry_align + np.pi) % (2 * np.pi) - np.pi
                
                # 无过滤，依然坚持对齐优先
                c = 0.2 * d + 3.0 * np.abs(h_diff) + 6.0 * np.abs(entry_align)
                
                if c < min_c:
                    min_c = c
                    bw = w
                    bidx = i
                
        return min_c, bw, bidx

    def _get_obs(self):
        lidar = self._get_lidar_readings() / self.lidar_range  # 归一化到 [0, 1]
        
        # 根据阶段选择目标点
        if self.phase == "exit":
            tp = self.exit_pos
            ty = self.exit_yaw
        else:
            tp = self.locked_w_pose['pos']
            ty = self.locked_w_pose['yaw']
        
        dx, dy = tp[0]-self.agent_pos[0], tp[1]-self.agent_pos[1]
        c, s = np.cos(self.agent_yaw), np.sin(self.agent_yaw)
        rx, ry = (dx*c + dy*s) / 10.0, (-dx*s + dy*c) / 10.0  # 归一化到约 [-1, 1]
        dyaw = (ty - self.agent_yaw + np.pi) % (2 * np.pi) - np.pi
        return np.concatenate([lidar, np.array([rx, ry], dtype=np.float32), np.array([np.sin(dyaw), np.cos(dyaw)], dtype=np.float32), self.prev_action]).astype(np.float32)

    def _get_lidar_readings(self):
        # 【修复】加上 agent_yaw，使雷达射线随车身旋转，索引 0 是车头正前方
        angles = np.linspace(0, 2*np.pi, self.n_lidar_rays, endpoint=False) + self.agent_yaw
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        p = self.agent_pos; a = self.obs_segments[:,0,:]; b = self.obs_segments[:,1,:]; v = b-a
        dx, dy = dirs[:,0][:,np.newaxis], dirs[:,1][:,np.newaxis]; vx, vy = v[:,0][np.newaxis,:], v[:,1][np.newaxis,:]
        det = -dx*vy + vx*dy; det[np.abs(det)<1e-9] = 1e-9
        rx, ry = a[:,0][np.newaxis,:]-p[0], a[:,1][np.newaxis,:]-p[1]
        t, u = (-rx*vy + vx*ry)/det, (dx*ry - dy*rx)/det
        valid = (t>0) & (u>=0) & (u<=1)
        return np.clip(np.min(np.where(valid, t, self.lidar_range), axis=1), 0.0, self.lidar_range)

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0
        
        if self.phase == "forward_rl":
            # ========== Stage 1: RL 导航到换向点 ==========
            v, delta = action
            v = np.clip(v, 0.0, 0.5)
            delta = np.clip(delta, -self.max_steer, self.max_steer)
            
            # 精确圆弧积分
            if np.abs(delta) > 1e-6:
                turning_radius = self.wheelbase / np.tan(delta)
                angular_velocity = v / turning_radius
                new_yaw = self.agent_yaw + angular_velocity
                new_yaw = (new_yaw + np.pi) % (2*np.pi) - np.pi
                dx = turning_radius * (np.sin(new_yaw) - np.sin(self.agent_yaw))
                dy = turning_radius * (np.cos(self.agent_yaw) - np.cos(new_yaw))
                self.agent_pos[0] += dx
                self.agent_pos[1] += dy
                self.agent_yaw = new_yaw
            else:
                self.agent_pos[0] += v * np.cos(self.agent_yaw)
                self.agent_pos[1] += v * np.sin(self.agent_yaw)
            self.traj_history.append(self.agent_pos.copy())
            
            # 基于距离触发的目标重选
            dist_to_site = np.linalg.norm(self.agent_pos - self.target_site['pos'])
            if not self.goal_switched and dist_to_site <= self.decision_dist:
                _, g_w, g_idx = self._calculate_global_best_w()
                self.locked_w_pose = g_w
                self.locked_idx = g_idx
                self.goal_switched = True
                self.prev_min_cost = self._compute_potential(self.agent_pos, self.agent_yaw, self.locked_w_pose)

            dist_err = np.linalg.norm(self.agent_pos - self.locked_w_pose['pos'])
            yaw_err = (self.agent_yaw - self.locked_w_pose['yaw'] + np.pi) % (2*np.pi) - np.pi
            
            # === "软惩罚"错过检测 (替代硬死刑) ===
            if self.goal_switched:
                if not self.approach_started:
                    self.approach_started = True
                    self.min_dist_to_w = dist_err
                else:
                    # 距离增大时给予柔性惩罚
                    if dist_err > self.min_dist_to_w:
                        overshoot_dist = dist_err - self.min_dist_to_w
                        reward -= 0.5 * overshoot_dist  # 柔性惩罚
                        self.overshoot_steps += 1
                    else:
                        self.min_dist_to_w = dist_err
                        self.overshoot_steps = 0  # 距离减小则重置
                    
                    # 只有连续超调 10 步且超过 1.0m (2倍车长) 才判定彻底失败
                    if self.overshoot_steps > 10 and dist_err > self.min_dist_to_w + 1.0:
                        reward = -80.0
                        terminated = True
                        self.current_step += 1
                        return self._get_obs(), reward, terminated, truncated, {'missed_target': True}
            
            curr_min_cost = self._compute_potential(self.agent_pos, self.agent_yaw, self.locked_w_pose)
            
            corners = self.get_rect_corners(self.agent_pos, self.agent_yaw, self.agent_length, self.agent_width)
            collision = (not self.is_rect_completely_inside(corners, self.boundary_coords) or 
                        self.is_rect_colliding_with_segments(corners, self.obs_segments))
            reached_switching = (dist_err < 0.25 and np.abs(yaw_err) < 0.40 and not collision)
            
            # Stage 1 奖励：大幅增加时间惩罚，降低速度奖励
            reward = (self.prev_min_cost - curr_min_cost) * 10.0
            reward += v * 0.1  # 降低速度奖励
            reward -= np.sum(np.maximum(0.35 - self._get_lidar_readings(), 0.0)) * 5.0
            reward -= 0.1      # 大幅增加时间惩罚 (从 0.01 增加到 0.1)
            if v < 0.1: reward -= 0.5
            reward -= (0.2*np.abs(v-self.prev_action[0]) + 0.3*np.abs(delta) + 1.5*np.abs(delta-self.prev_action[1]) + 0.5*delta**2)

            if dist_err < 0.15 and not self.has_reached_pos and not collision:
                reward += 20.0
                self.has_reached_pos = True

            if collision:
                reward = -100.0
                terminated = True
            elif reached_switching:
                # 到达换向点，切换到 Stage 2
                reward += 50.0  # Stage 1 完成奖励
                self.phase = "reversing"
                self.arc_progress = 0.0
                # 精确对齐到换向点
                self.agent_pos = self.locked_w_pose['pos'].copy()
                self.agent_yaw = self.locked_w_pose['yaw']
            
            self.prev_min_cost = curr_min_cost
            self.prev_action = action.copy()
        
        elif self.phase == "reversing":
            # ========== Stage 2: 确定性倒车沿曲线 ==========
            self.arc_progress += 0.02  # 每步前进 2%，约 50 步完成
            self._move_along_arc(self.arc_progress)
            self.traj_history.append(self.agent_pos.copy())
            
            # 检查碰撞
            corners = self.get_rect_corners(self.agent_pos, self.agent_yaw, self.agent_length, self.agent_width)
            collision = (not self.is_rect_completely_inside(corners, self.boundary_coords) or 
                        self.is_rect_colliding_with_segments(corners, self.obs_segments))
            
            if collision:
                reward = -100.0
                terminated = True
            elif self.arc_progress >= 1.0:
                # 到达排土点，切换到 Stage 3 出库
                reward += 50.0  # Stage 2 完成奖励
                self.phase = "exit"
                self.exit_progress = 0.0
                # 精确对齐到排土点
                self.agent_pos = self.target_site['pos'].copy()
                self.agent_yaw = self.target_site['yaw']
            else:
                reward += 1.0  # 每步给小奖励鼓励继续
        
        elif self.phase == "exit":
            # ========== Stage 3: RL 出库导航 ==========
            v, delta = action
            v = np.clip(v, 0.0, 0.5)
            delta = np.clip(delta, -self.max_steer, self.max_steer)
            
            # 精确圆弧积分
            if np.abs(delta) > 1e-6:
                turning_radius = self.wheelbase / np.tan(delta)
                angular_velocity = v / turning_radius
                new_yaw = self.agent_yaw + angular_velocity
                new_yaw = (new_yaw + np.pi) % (2*np.pi) - np.pi
                dx = turning_radius * (np.sin(new_yaw) - np.sin(self.agent_yaw))
                dy = turning_radius * (np.cos(self.agent_yaw) - np.cos(new_yaw))
                self.agent_pos[0] += dx
                self.agent_pos[1] += dy
                self.agent_yaw = new_yaw
            else:
                self.agent_pos[0] += v * np.cos(self.agent_yaw)
                self.agent_pos[1] += v * np.sin(self.agent_yaw)
            self.traj_history.append(self.agent_pos.copy())
            
            # 计算到出口的距离和角度误差
            dist_to_exit = np.linalg.norm(self.agent_pos - self.exit_pos)
            yaw_to_exit = np.arctan2(self.exit_pos[1] - self.agent_pos[1], 
                                      self.exit_pos[0] - self.agent_pos[0])
            yaw_err = (self.agent_yaw - yaw_to_exit + np.pi) % (2*np.pi) - np.pi
            
            # 检查碰撞
            corners = self.get_rect_corners(self.agent_pos, self.agent_yaw, self.agent_length, self.agent_width)
            collision = (not self.is_rect_completely_inside(corners, self.boundary_coords) or 
                        self.is_rect_colliding_with_segments(corners, self.obs_segments))
            
            reached_exit = (dist_to_exit < 0.3 and not collision)  # 出口阈值稍宽松
            
            # Stage 3 奖励：增加姿态引导，防止冲过头
            # 定义 Stage 3 的目标 Pose
            exit_target_pose = {'pos': self.exit_pos, 'yaw': self.exit_yaw}
            
            # 使用统一势能函数计算进度奖励
            curr_exit_cost = self._compute_potential(self.agent_pos, self.agent_yaw, exit_target_pose)
            prev_exit_cost = getattr(self, 'prev_exit_cost', curr_exit_cost)
            
            reward = (prev_exit_cost - curr_exit_cost) * 15.0  # 势能下降奖励
            reward += v * 0.1   # 降低速度奖励
            reward -= 0.2       # 增加时间惩罚 (出库阶段惩罚更高，促使迅速离开)
            if v < 0.1: reward -= 0.3
            self.prev_exit_cost = curr_exit_cost
            
            if collision:
                reward = -100.0
                terminated = True
            elif reached_exit:
                # 到达出口，任务完全完成
                reward += 200.0
                terminated = True
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _move_along_arc(self, progress):
        """沿预采样曲线移动到指定进度位置（0=换向点，1=排土点）"""
        progress = np.clip(progress, 0.0, 1.0)
        L = self.locked_w_pose['L']
        dt = self.locked_w_pose['delta_theta']
        r = self.locked_w_pose['r']
        
        pos_d = self.target_site['pos']
        yaw_d = self.target_site['yaw']
        
        # 反向进度：从排土点出发，progress=0 对应换向点
        arc_t = (1.0 - progress) * dt
        
        if np.abs(dt) < 1e-6:
            # 直线
            dist_along = (1.0 - progress) * L
            self.agent_pos = pos_d + np.array([np.cos(yaw_d), np.sin(yaw_d)]) * dist_along
            self.agent_yaw = yaw_d
        else:
            # 圆弧
            rs = np.sign(dt)
            xl = r * np.sin(np.abs(arc_t))
            yl = rs * r * (1.0 - np.cos(np.abs(arc_t)))
            c, s = np.cos(yaw_d), np.sin(yaw_d)
            self.agent_pos = pos_d + np.array([c * xl - s * yl, s * xl + c * yl])
            self.agent_yaw = yaw_d + arc_t

    def _move_to_exit(self, progress):
        """从排土点直线移动到出口点"""
        progress = np.clip(progress, 0.0, 1.0)
        start_pos = self.target_site['pos']
        start_yaw = self.target_site['yaw']
        
        # 线性插值位置
        self.agent_pos = start_pos + progress * (self.exit_pos - start_pos)
        
        # 逐渐转向出口方向
        yaw_diff = self.exit_yaw - start_yaw
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi  # 归一化
        self.agent_yaw = start_yaw + progress * yaw_diff

    def get_rect_corners(self, pos, yaw, length, width):
        c, s = np.cos(yaw), np.sin(yaw); rot = np.array([[c, -s], [s, c]])
        half_l, half_w = length / 2.0, width / 2.0
        corners = np.array([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]])
        return corners @ rot.T + pos

    def is_point_in_polygon(self, point, polygon):
        x, y = point; n = len(polygon); inside = False; p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y: xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                if p1x == p2x or x <= xinters: inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def is_rect_completely_inside(self, corners, polygon):
        for pt in corners:
            if not self.is_point_in_polygon(pt, polygon): return False
        return True

    def is_rect_colliding_with_segments(self, corners, segments):
        rect_edges = [[corners[i], corners[(i+1)%4]] for i in range(4)]
        for edge in rect_edges:
            for seg in segments:
                if self._line_segments_intersect(edge[0], edge[1], seg[0], seg[1]): return True
        return False

    def _line_segments_intersect(self, p1, p2, p3, p4):
        def ccw(A, B, C): return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

    def is_path_safe(self, w_pose, d_pose, dilation=0.2):
        L, dt, r = w_pose['L'], w_pose['delta_theta'], w_pose['r']
        pos_d, yaw_d = d_pose['pos'], d_pose['yaw']
        n_samples = max(int(L / 0.1), 2); dil_l, dil_w = self.agent_length + dilation, self.agent_width + dilation
        sc, ss = np.cos(yaw_d), np.sin(yaw_d)
        for i in range(n_samples + 1):
            ratio = i / n_samples
            if np.abs(dt) < 1e-6: cp = pos_d + np.array([sc, ss]) * (ratio * L); cy = yaw_d
            else:
                t = ratio * dt; rs = np.sign(dt); xl, yl = r * np.sin(np.abs(t)), rs * r * (1.0 - np.cos(np.abs(t)))
                cp = pos_d + np.array([sc * xl - ss * yl, ss * xl + sc * yl]); cy = yaw_d + t
            corners = self.get_rect_corners(cp, cy, dil_l, dil_w)
            if not self.is_rect_completely_inside(corners, self.boundary_coords): return False
            if self.is_rect_colliding_with_segments(corners, self.obs_segments): return False
        return True

    def sample_dumping_sites(self, interval=1.5):
        boundary_segments = self.obstacle_lines[:11]; sites = []; next_sample_dist = 0.0; temp_d_info = []
        for seg in boundary_segments:
            p1, p2 = seg[0], seg[1]; vec = p2 - p1; length = np.linalg.norm(vec)
            while next_sample_dist <= length:
                ratio = next_sample_dist / length; unit_inward_normal = np.array([-vec[1], vec[0]]) / length
                yaw = np.arctan2(unit_inward_normal[1], unit_inward_normal[0])
                pos = (p1 + ratio * vec) + 0.55 * unit_inward_normal
                if pos[1] <= 6.0: temp_d_info.append({'pos': pos, 'yaw': yaw})
                next_sample_dist += interval
            next_sample_dist -= length
        L_fixed = np.pi * self.min_turning_radius / 2.0
        z_deg_list = [0, 10, -10, 20, -20, 30, -30, 40, -40, 45, -45, 50, -50, 60, -60, 70, -70, 80, -80, 90, -90]
        
        # 预先收集所有 D 点位置，用于 W 点避让校验
        all_d_poses = [d['pos'] for d in temp_d_info]
        
        for d_site in temp_d_info:
            w_poses = []
            for z_deg in z_deg_list:
                w_pose = self.calculate_arc_switching_pose(d_site, L=L_fixed, delta_theta=np.radians(z_deg))
                
                # 校验 1: W 点必须离开所有的 D 点 (除了自己的 D 点) 至少 1.2m
                is_overlap_with_d = False
                for other_d_pos in all_d_poses:
                    if np.array_equal(other_d_pos, d_site['pos']): continue
                    if np.linalg.norm(w_pose['pos'] - other_d_pos) < 0.6:
                        is_overlap_with_d = True
                        break
                if is_overlap_with_d: continue
                
                # 校验 2: 路径安全 (避障)
                if self.is_path_safe(w_pose, d_site, dilation=0.2): 
                    w_poses.append(w_pose)
                    
            if len(w_poses) > 0: 
                sites.append({'pos': d_site['pos'], 'yaw': d_site['yaw'], 'w_poses': w_poses})
        return sites

    def calculate_arc_switching_pose(self, d_pose, L, delta_theta, r_min=1.25):
        x_d, y_d = d_pose['pos']; theta_d = d_pose['yaw']
        if np.abs(delta_theta) < 1e-6: xl, yl, tl, r = L, 0.0, 0.0, np.inf
        else:
            r = max(L / np.abs(delta_theta), r_min); dt = np.sign(delta_theta)*(L/r) if r==r_min else delta_theta
            xl, yl, tl = r * np.sin(np.abs(dt)), np.sign(dt) * r * (1.0 - np.cos(np.abs(dt))), dt
        c, s = np.cos(theta_d), np.sin(theta_d)
        return {'pos': np.array([x_d+c*xl-s*yl, y_d+s*xl+c*yl]), 'yaw': theta_d+tl, 'L': L, 'delta_theta': delta_theta, 'r': r}

    def save_training_frame(self, save_path):
        data = self.last_episode_data if self.last_episode_data else {
            'traj': self.traj_history, 'target_site': self.target_site, 'locked_w_pose': self.locked_w_pose,
            'agent_pos': self.agent_pos, 'agent_yaw': self.agent_yaw, 'final_step': self.current_step,
            'goal_switched': self.goal_switched, 'phase': self.phase,
            'exit_pos': self.exit_pos.copy()
        }
        fig, ax = plt.subplots(figsize=(10, 10))
        for seg in self.obs_segments: ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color='blue', linewidth=2, alpha=0.5)
        dl, dw = 0.7, 0.4; c_target = data['target_site']; locked_w = data['locked_w_pose']
        for site in self.dumping_sites:
            pos, yaw = site['pos'], site['yaw']; is_target = np.array_equal(pos, c_target['pos'])
            sc, ss = np.cos(yaw), np.sin(yaw); rot = np.array([[sc, -ss], [ss, sc]])
            corners_rect = np.array([[dl/2, dw/2], [dl/2, -dw/2], [-dl/2, -dw/2], [-dl/2, dw/2], [dl/2, dw/2]])
            rotated_corners = corners_rect @ rot.T + pos
            color = 'orange' if is_target else 'green'; alpha_s = 0.9 if is_target else 0.3
            ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], color=color, linewidth=2.5 if is_target else 1.0, alpha=alpha_s)
            ax.fill(rotated_corners[:, 0], rotated_corners[:, 1], color=color, alpha=0.1)
            ax.arrow(pos[0], pos[1], sc * 0.2, ss * 0.2, head_width=0.03, head_length=0.06, fc=color, ec=color, alpha=alpha_s)
            if is_target:
                ax.plot(pos[0], pos[1], marker='*', markersize=15, color='gold', markeredgecolor='orange', zorder=15)
                w_pos, w_yaw = locked_w['pos'], locked_w['yaw']; wc, ws = np.cos(w_yaw), np.sin(w_yaw)
                w_rot = np.array([[wc, -ws], [ws, wc]]); r_w = corners_rect @ w_rot.T + w_pos
                ax.plot(r_w[:, 0], r_w[:, 1], color='red', linewidth=2.5, alpha=1.0)
                dt_v, rv = locked_w['delta_theta'], locked_w['r']
                if np.abs(dt_v) < 1e-6: ax.plot([pos[0], w_pos[0]], [pos[1], w_pos[1]], 'k--', linewidth=0.5, alpha=0.3)
                else:
                    ts = np.linspace(0, dt_v, 15); rs_val = np.sign(dt_v); xa, ya = rv*np.sin(np.abs(ts)), rs_val*rv*(1.0-np.cos(np.abs(ts)))
                    xg, yg = pos[0]+sc*xa-ss*ya, pos[1]+ss*xa+sc*ya
                    ax.plot(xg, yg, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
        trag = np.array(data['traj'])
        if len(trag) > 1:
            ax.plot(trag[:, 0], trag[:, 1], color='magenta', linewidth=1.5, alpha=0.6)
        a_pos, a_yaw = data['agent_pos'], data['agent_yaw']; ca, sa = np.cos(a_yaw), np.sin(a_yaw)
        rot_aa = np.array([[ca, -sa], [sa, ca]])
        corners_a = np.array([[self.agent_length/2, self.agent_width/2], [self.agent_length/2, -self.agent_width/2], [-self.agent_length/2, -self.agent_width/2], [-self.agent_length/2, self.agent_width/2], [self.agent_length/2, self.agent_width/2]])
        r_c_aa = corners_a @ rot_aa.T + a_pos
        ax.plot(r_c_aa[:, 0], r_c_aa[:, 1], color='purple', linewidth=2); ax.arrow(a_pos[0], a_pos[1], ca*0.4, sa*0.4, head_width=0.08, head_length=0.12, fc='purple', ec='purple')
        ax.set_xlim(-1, 11); ax.set_ylim(-1, 11); ax.set_aspect('equal'); ax.grid(True, linestyle=':', alpha=0.5)
        # 绘制出口点
        if 'exit_pos' in data:
            exit_p = data['exit_pos']
            ax.plot(exit_p[0], exit_p[1], marker='s', markersize=12, color='cyan', markeredgecolor='blue', zorder=15, label='Exit')
        phase_text = data.get('phase', 'forward_rl')
        if phase_text == "forward_rl":
            phase_display = "Stage1:RL"
        elif phase_text == "reversing":
            phase_display = "Stage2:Reversing"
        else:
            phase_display = "Stage3:Exit"
        sw_text = "(Final Decided)" if data['goal_switched'] else "(Early Prep)"
        ax.set_title(f"{phase_display} {sw_text} | Steps {data['final_step']}")
        if save_path: plt.savefig(save_path, dpi=100); plt.close(fig)
        else: plt.show()

