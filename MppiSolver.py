import numpy as np
import math

class MppiplanSolver:
    """局部规划求解器（标准MPPI实现，适配 [linear_vel, angular_vel] 控制输入）"""
    
    def __init__(self, x0, xf, xg, obstacles=None):
        self.dim_x = 3  # 状态维度 [x, y, yaw]
        self.dim_u = 2  # 控制维度 [linear_vel, angular_vel]
        self.T = 20     # 预测时域长度（步数）
        self.K = 500    # 采样数量K（生成K条轨迹）
        self.param_lambda = 0.5  # 温度参数（控制权重分布陡峭度）
        self.param_alpha = 0.0   # 标准MPPI通常不使用带衰减的记忆更新

        # --- 关键调整 1: 噪声协方差 sigma ---
        # 针对 [linear_vel, angular_vel] 调整
        # 初始值设小一些，避免过大的随机探索
        # 示例: 线速度标准差 0.5 m/s, 角速度标准差 10 deg/s
        self.sigma = np.array([[0.5**2, 0.0], [0.0, np.deg2rad(10.0)**2]]) 
        # -----------------------------------

        # --- 关键调整 2: 成本权重 ---
        # 增加对控制输入的惩罚，平衡状态追踪
        self.stage_cost_weight = np.array([20.0, 20.0, 10.0])  # [x, y, theta] - 可根据需要调整
        self.input_cost_weight = np.array([1.0, 0.5])          # [V, W] - 增大以抑制过大输入
        self.terminal_cost_weight = np.array([50.0, 50.0, 20.0]) # [x, y, theta] - 终端更严格
        # --------------------------

        # --- 保留但可能未使用的参数 ---
        self.max_search_idx_len = 50
        self.obstacle_cost_weight = 50.0
        self.safety_margin = 1.0
        self.max_repulsive_force = 15.0
        # -----------------------------

        # --- 关键调整 3: 控制输入限制 (对应 [V, W]) ---
        # 根据你的系统物理限制设置
        self.max_linear_vel_abs = 2.0  # m/s (示例，请根据你的机器人/车辆调整)
        self.max_angular_vel_abs = np.deg2rad(45.0) # rad/s (示例，请根据你的机器人/车辆调整)
        # ---------------------------------------------

        self.u_prev = np.zeros((self.T, self.dim_u)) # 历史控制序列

        # --- 保留但可能未使用的参数 ---
        self.pre_waypoints_idx = 0
        self.pi = 3.1415926
        self.wheel_base = 2.5 # 如果 _next_x 用到，需要调整
        # -----------------------------

        self.x0 = np.array(x0)
        self.xf = np.array(xf) # 目标状态
        self.xg = np.array(xg) # 可能是全局路径点或目标点

        self.obstacles = obstacles
        self.delta_t = 0.1 # 时间步长
        
        # 求解结果
        self.trajectory = None
        self.solver_result = None
        self.cost = None

    def calc_control_input(self, x0=None, xf=None, xg=None):
        """计算控制输入 (标准MPPI流程)"""
        # 1. 使用上一时刻的控制序列作为基础 (状态前馈)
        u = self.u_prev.copy() 

        # 更新内部状态（如果外部提供了）
        if x0 is not None:
            self.x0 = np.array(x0)
        if xf is not None:
            self.xf = np.array(xf)
        if xg is not None:
            self.xg = np.array(xg)

        x0 = self.x0 # 使用当前内部状态

        # 2. 初始化成本和噪声
        S = np.zeros(self.K) # 每条轨迹的成本
        epsilon = self._calc_epsilon(self.sigma, self.K, self.T, self.dim_u) # 生成噪声

        # 3. 采样与前向模拟 (Rollout)
        v = np.zeros((self.K, self.T, self.dim_u))
        sampled_traj_list = np.zeros((self.K, self.T, self.dim_x)) # 存储所有采样轨迹

        for k in range(self.K):
            x = x0.copy() # 从当前状态开始
            for t in range(self.T):
                # 添加噪声到名义控制输入
                v[k, t] = u[t] + epsilon[k, t]
                
                # --- 关键点: 成本计算使用未裁剪的 v ---
                S[k] += self._stage_cost(x) + self._input_cost(v[k, t]) 
                # -------------------------------------
                
                # --- 关键点: 限制控制输入用于模拟 (防止物理上不可能的状态) ---
                v_clamped = self._u_clamp(v[k, t].copy()) 
                # -----------------------------------------------------------
                
                # 前向模拟一步
                x = self._next_x(x, v_clamped)
                sampled_traj_list[k, t] = x.copy() # 存储模拟的状态
            
            # 累积终端成本
            S[k] += self._terminal_cost(x)

        # 4. 计算权重
        w = self._calc_weights(S)

        # 5. 更新控制序列 (标准MPPI更新)
        u_new = np.zeros_like(u)
        for t in range(self.T):
             weighted_epsilon_sum = np.zeros(self.dim_u)
             for k in range(self.K):
                 weighted_epsilon_sum += w[k] * epsilon[k, t]
             u_new[t] = u[t] + weighted_epsilon_sum 

        # 6. 滚动更新 u_prev
        self.u_prev[:-1] = u_new[1:]
        self.u_prev[-1] = u_new[-1] # 末尾保持最后一个控制

        # 7. 计算(近似)最优轨迹用于返回 (使用更新后的控制序列)
        optimal_traj = np.zeros((self.T, self.dim_x))
        x = x0.copy()
        for t in range(self.T):
            u_clamped = self._u_clamp(u_new[t].copy())
            x = self._next_x(x, u_clamped)
            optimal_traj[t] = x

        xy = optimal_traj[:, :2]  # 提取 x, y 坐标

        # 返回值保持接口一致
        return u_new[0], u_new, xy, sampled_traj_list

    def _calc_weights(self, costs):
        """计算各采样轨迹的权重 (标准MPPI)"""
        min_cost = np.min(costs)
        # 提高数值稳定性
        exp_terms = np.exp((-1.0 / self.param_lambda) * (costs - min_cost))
        weights = exp_terms / (np.sum(exp_terms) + 1e-10) # 防止除零
        return weights

    def _terminal_cost(self, x_T):
        """计算终端成本"""
        x, y, theta = x_T
        dx = x - self.xf[0]
        dy = y - self.xf[1]
        dtheta = self.WrapToPi(theta - self.xf[2])
        terminal_cost = (self.terminal_cost_weight[0] * dx**2 +
                         self.terminal_cost_weight[1] * dy**2 +
                         self.terminal_cost_weight[2] * dtheta**2)
        return terminal_cost

    def _stage_cost(self, x_t):
        """计算阶段成本 (到目标状态 xf 的偏差)"""
        x, y, theta = x_t
        dx = x - self.xf[0]
        dy = y - self.xf[1]
        dtheta = self.WrapToPi(theta - self.xf[2]) # 角度差需要特殊处理
        stage_cost = (self.stage_cost_weight[0] * dx**2 +
                      self.stage_cost_weight[1] * dy**2 +
                      self.stage_cost_weight[2] * dtheta**2)
        return stage_cost

    def _input_cost(self, u_t):
        """计算控制输入成本 (惩罚控制量大小)"""
        # 假设权重是向量，与 u_t 对应元素相乘
        input_cost = self.input_cost_weight[0] * u_t[0]**2 + self.input_cost_weight[1] * u_t[1]**2
        return input_cost

    def _next_x(self, x_t: np.ndarray, u_t: np.ndarray) -> np.ndarray:
        """系统动力学模型 (差分驱动/简单移动机器人模型)"""
        x_t = np.asarray(x_t).ravel()
        u_t = np.asarray(u_t).ravel()
        
        linear_vel, angular_vel = u_t[0], u_t[1] # u[0]是线速度V, u[1]是角速度W
        x, y, theta = x_t

        # 简单的欧拉积分差分驱动模型
        new_x = x + linear_vel * np.cos(theta) * self.delta_t
        new_y = y + linear_vel * np.sin(theta) * self.delta_t
        new_theta = theta + angular_vel * self.delta_t
        new_theta = self.WrapToPi(new_theta) # 角度归一化很重要

        return np.array([new_x, new_y, new_theta]).reshape(x_t.shape)

    def _u_clamp(self, u):
        """限制控制输入在可行范围内 ([V, W])"""
        # 使用为 [V, W] 定义的限制值
        u[0] = np.clip(u[0], -self.max_linear_vel_abs, self.max_linear_vel_abs) # 限制线速度 V
        u[1] = np.clip(u[1], -self.max_angular_vel_abs, self.max_angular_vel_abs) # 限制角速度 W
        return u

    def _calc_epsilon(self, sigma, K, T, dim_u):
        """生成高斯噪声序列 (标准MPPI核心)"""
        return np.random.multivariate_normal(np.zeros(dim_u), sigma, (K, T))

    def WrapToPi(self, rad: float) -> float:
        """将角度转换到 [-pi, pi] 范围内 (使用NumPy更高效)"""
        res = (rad + np.pi) % (2 * np.pi) - np.pi
        return res


# --- 简单测试代码 (可选) ---
# if __name__ == '__main__':
#     # 初始状态 [x, y, theta]
#     x0 = np.array([0.0, 0.0, 0.0])
#     # 目标状态 [x, y, theta]
#     xf = np.array([5.0, 3.0, np.pi/2])
#     # 全局目标点 (如果需要)
#     xg = np.array([5.0, 3.0])
#
#     solver = MppiplanSolver(x0, xf, xg)
#
#     # 模拟几步看看控制输出
#     for i in range(5):
#         print(f"\n--- Iteration {i} ---")
#         u_first, u_seq, traj_xy, sampled_trajs = solver.calc_control_input()
#         print(f"First control input (V, W): {u_first}")
#         print(f"Current state estimate: {solver.x0}")
#         # 假设应用了控制并前进了
#         # 这里模拟状态更新 (实际应用中由传感器/定位提供)
#         # next_state = solver._next_x(solver.x0, u_first)
#         # solver.x0 = next_state
#         # print(f"Next state estimate: {next_state}")
#         
#         # 检查一下权重分布，看看是否有明显的最优轨迹
#         # (需要访问内部S或修改代码返回S)
#         # print(f"Min/Max Cost: {np.min(solver.S)}, {np.max(solver.S)}") 
#         # print(f"Max Weight: {np.max(solver.w)}")




