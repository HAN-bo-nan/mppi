import numpy as np
from irsim.env import EnvBase
from MppiSolver import MppiplanSolver
from irsim.lib.path_planners.a_star import AStarPlanner
from matplotlib import pyplot as plt
import sys
import math


class SIM_ENV:
    def __init__(self, world_file="robot_world.yaml", render=False):

        # 初始化环境
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render,save_ani = True)
        # 环境参数
        self.robot_goal = self.env.get_robot_info(0).goal
        self.lidar_r = 1.0
        data = self.env.get_map()
        # # 全局规划器
        start = self.env.get_robot_state().T
        start = start[0, :2].squeeze()
        end = self.robot_goal.T
        end = end[0, :2].squeeze()

        data = self.env.get_map()

        npy_path = sys.path[0] + '/path_track_ref.npy'
        ref_path_list = list(np.load(npy_path, allow_pickle=True))
        self.env.draw_trajectory(ref_path_list, traj_type='-k')
        self.ref_path_list = np.array(ref_path_list)
        # print(self.ref_path_list)

        # self.planner = AStarPlanner(data, data.resolution)
        # self.global_path = self.planner.planning(start, end, show_animation=False)
        # self.global_path = self.global_path[:, ::-1].T

        self.path_index = 0



        self.delta_t = 0.1
        self.pi = 3.1415926
        # 局部求解器
        self.solver = MppiplanSolver(np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0,0.0]),[8.5, 1.5])
        # 速度指令
        self.v = 0.0
        self.w = 0.0
        self.robot_state = [[1.5],[8.5],[0.],[0.]]
        self.robot_state = np.array(self.robot_state)
        # self.env.draw_trajectory(traj=self.global_path.T, traj_type="--y")
    def step(self,):
        # 环境单步仿真
        self.env.step(action_id=0, action=np.array([[self.v], [self.w]]))
        # 环境可视化
        if self.env.display:
            self.env.render()
        self.robot_state = self.env.get_robot_state()
        scan_data = self.env.get_lidar_scan()
        obs_list, center_list = self.scan_box(self.robot_state, scan_data)

        for obs in obs_list:
            self.env.draw_box(obs, refresh=True, color="-b")
        
        # 计算临时目标点
        current_goal= self.find_target_point(self.robot_state)
        print(current_goal)
        self.env.draw_points(current_goal[:2], c="r", refresh=True)
        # 求解局部最优轨迹
        optimal_input, _,xy, sampled_traj_list= self.solver.calc_control_input(self.robot_state.squeeze(),current_goal.squeeze(),[8.5, 8.5])
        self.v = optimal_input[0]
        self.w = optimal_input[1]
        self.update(self.robot_state,optimal_input)
        traj_list = [np.array([[x], [y]]) for x, y in xy]
        self.env.draw_trajectory(traj_list, 'r', refresh=True)
        xy_3d = sampled_traj_list[:, :, :2] 
        traj_list_sam = []                       
        for traj in xy_3d:                           
            traj_list_sam.append([np.array([[x], [y]]) for x, y in traj])
        for traj in traj_list_sam:
            self.env.draw_trajectory(traj, 'b-', linewidth=0.5, refresh=True)

        if self.env.robot.arrive:
            print("Goal reached")
            return True

        # 是否碰撞
        if self.env.robot.collision:
            print("collision !!!")
            return True
        
        return False
    
    def find_target_point(self, robot_state):
        """
        顺序发点版，直接替换原函数
        robot_state : [x, y, theta]
        """
        # 第一次调用时初始化
        if not hasattr(self, '_seq_idx'):
            self._seq_idx = 0          # 当前要追的点的索引
            self._reach_R = 0.80       # 到达阈值，可调

        path = self.ref_path_list          # (N,3)
        N = len(path)

        # 如果已经走完，永远返回最后一个点
        if self._seq_idx >= N - 1:
            return path[-1]

        # 计算到当前点的距离
        dx = path[self._seq_idx, 0] - robot_state[0]
        dy = path[self._seq_idx, 1] - robot_state[1]
        if np.hypot(dx, dy) < self._reach_R:
            self._seq_idx += 1
            # 切换后保护越界
            if self._seq_idx >= N:
                self._seq_idx = N - 1

        return path[self._seq_idx]
    

    def update(self, robot_state: np.ndarray, v_t: list) -> np.ndarray:

        # 统一转 1-D 数组，最后再 reshape 回去
        robot_state   = np.asarray(robot_state).ravel()
        linear, angular = v_t[0], v_t[1]
        x, y, theta = robot_state

        # 3. 运动学
        robot_state[0]  = x + linear * np.cos(theta) * self.delta_t
        robot_state[1]  = y + linear * np.sin(theta) * self.delta_t
        new_theta = theta + angular * self.delta_t
        robot_state[2] = self.WrapToPi(new_theta)

    def WrapToPi(self ,rad: float, positive: bool = False) -> float:
        """The function `WrapToPi` transforms an angle in radians to the range [-pi, pi].

        Args:

            rad (float): Angle in radians. The `rad` parameter in the `WrapToPi` function represents an angle in radians that you want to transform to the range `[-π, π]`. The function ensures that the angle is within this range by wrapping it around if it exceeds the bounds.

            positive (bool): Whether to return the positive value of the angle. Useful for angles difference.

        Returns:
            The function `WrapToPi(rad)` returns the angle `rad` wrapped to the range [-pi, pi].

        """
        while rad > self.pi:
            rad = rad - 2 * self.pi
        while rad < -self.pi:
            rad = rad + 2 * self.pi

        return rad if not positive else abs(rad)
    
    def scan_box(self, state, scan_data):

        ranges = np.array(scan_data['ranges'])
        angles = np.linspace(scan_data['angle_min'], scan_data['angle_max'], len(ranges))

        point_list = []
        obstacle_list = []
        center_list = []

        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]

            if scan_range < (scan_data['range_max'] - 0.1):
                point = np.array([[scan_range * np.cos(angle)], [scan_range * np.sin(angle)]])
                point_list.append(point)

        if len(point_list) < 4:
            return obstacle_list, center_list

        else:
            point_array = np.hstack(point_list).T
            labels = DBSCAN(eps=0.2, min_samples=2).fit_predict(point_array)

            for label in np.unique(labels):
                if label == -1:
                    continue
                else:
                    point_array2 = point_array[labels == label]
                    rect = cv2.minAreaRect(point_array2.astype(np.float32))
                    box = cv2.boxPoints(rect)
                    center_local = np.array(rect[0]).reshape(2, 1)

                    vertices = box.T

                    trans = state[0:2]
                    rot = state[2, 0]
                    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
                    global_vertices = trans + R @ vertices
                    center_global = trans + R @ center_local

                    obstacle_list.append(global_vertices)
                    center_list.append(center_global)

            return obstacle_list, center_list
        
    






