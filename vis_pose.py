import numpy as np
import pandas as pd
import open3d as o3d
from matplotlib.colors import hsv_to_rgb
import os
import time

# ================== 配置参数 ==================
data_dir = "20250820-212141"
ply_path = os.path.join(data_dir, "mesh.ply")
extrinsics_csv = os.path.join(data_dir, "extrinsics_log.csv")
intrinsics_csv = os.path.join(data_dir, "intrinsics_log.csv")

# 相机参数
width, height = 1920, 1080          # 图像分辨率
far_plane = 0.1                     # 视锥体远平面距离（可调）
scale_current = 2.0                 # 当前帧视锥体放大倍数
history_length = 10                 # 显示最近 N 个历史视锥体
frame_delay = 0.01                   # 每帧间隔（秒），控制播放速度
show_coordinate_frames = True       # 是否显示每个相机的坐标系（调试用）

# ================== 工具函数 ==================
def read_ply(filepath):
    """读取PLY网格"""
    mesh = o3d.io.read_triangle_mesh(filepath)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh

def read_extrinsics(csv_file):
    """读取外参：T_cam_from_world (4x4)"""
    df = pd.read_csv(csv_file)
    timestamps = df['timestamp'].values
    extrinsics = []
    for _, row in df.iterrows():
        mat = np.array(row[1:17]).reshape(4, 4)
        extrinsics.append(mat)
    return timestamps, extrinsics

def read_intrinsics(csv_file):
    """读取内参：K (3x3)"""
    df = pd.read_csv(csv_file)
    intrinsics = []
    for _, row in df.iterrows():
        fx = row['m00']
        fy = row['m11']
        cx = row['m02']
        cy = row['m12']
        K = np.array([[fx,  0, cx],
                      [ 0, fy, cy],
                      [ 0,  0,  1]])
        intrinsics.append(K)
    return intrinsics

def create_frustum_lines(K, width, height, far, T_world_from_cam, scale=1.0):
    """
    在相机坐标系中构建视锥体，并通过 T_world_from_cam 变换到世界坐标
    返回：points (N,3), lines (M,2)
    """
    # 相机坐标系下视锥顶点（Z向前）
    x_right = far * (width / 2) / K[0, 0]
    x_left = far * (-width / 2) / K[0, 0]
    y_top = far * (-height / 2) / K[1, 1]
    y_bottom = far * (height / 2) / K[1, 1]

    points_cam = np.array([
        [0, 0, 0],              # 相机中心
        [x_left,  y_top,    far],  # 远平面四角
        [x_right, y_top,    far],
        [x_right, y_bottom, far],
        [x_left,  y_bottom, far]
    ]) * scale

    # 变换到世界坐标
    points_world = []
    for pt in points_cam:
        pt_h = np.append(pt, 1.0)  # 齐次坐标
        pt_world = T_world_from_cam @ pt_h
        points_world.append(pt_world[:3])
    points_world = np.array(points_world)

    # 线段：中心到四角 + 远平面四边
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # 中心到四角
        [1, 2], [2, 3], [3, 4], [4, 1]   # 远平面框
    ]
    return points_world, lines

def get_correction_transform():
    """
    返回一个 4x4 变换矩阵 T_AC，表示：
    从 ARKit 相机坐标系 → 校正后坐标系（绕 X 轴旋转 180°）
    等价于 Swift 中的 q_AC = simd_quatf(ix: 1.0, iy: 0.0, iz: 0.0, r: 0.0)
    """
    T_AC = np.eye(4)
    # 绕 X 轴旋转 180°
    T_AC[1,1] = -1  # Y → -Y
    T_AC[2,2] = -1  # Z → -Z
    return T_AC

# ================== 自动播放动画类 ==================
class AutoFrustumAnimator:
    def __init__(self, mesh, extrinsics, intrinsics):
        self.mesh = mesh
        self.extrinsics = extrinsics
        self.intrinsics = intrinsics
        self.n_frames = len(extrinsics)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.mesh)

        # 存储历史视锥体
        self.history_linesets = []

        # 颜色：按时间渐变（蓝→红）
        self.colors = [hsv_to_rgb([i / max(self.n_frames - 1, 1) * 0.7, 0.8, 0.8]) 
                       for i in range(self.n_frames)]

        # 坐标系对齐矩阵：将原始世界坐标系（Z-up）→ Y-up
        self.T_align = np.eye(4)
        self.T_align[:3, :3] = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0]
        ])

        # 当前视锥体
        self.current_lineset = None
        self.current_frame = None  # 坐标系frame

    def run(self):
        print(f"开始自动播放，共 {self.n_frames} 帧，每帧间隔 {frame_delay}s")
        print("可在播放过程中自由旋转/缩放视角，ESC 退出")

        for idx in range(self.n_frames):
            start_time = time.time()

            # ========== 清除上一帧 ==========
            if self.current_lineset is not None:
                self.vis.remove_geometry(self.current_lineset, reset_bounding_box=False)
            if self.current_frame is not None:
                self.vis.remove_geometry(self.current_frame, reset_bounding_box=False)

            for ls in self.history_linesets:
                self.vis.remove_geometry(ls, reset_bounding_box=False)
            self.history_linesets.clear()

            # 获取当前帧数据
            T_cam_from_world = self.extrinsics[idx]  # world → cam
            T_AC = get_correction_transform()
            T_cam_from_world = T_cam_from_world @ T_AC  # 应用校正变换
            
            
            K = self.intrinsics[idx]
            color = self.colors[idx]
            

            T_world_from_cam = T_cam_from_world
            points_curr, lines_curr = create_frustum_lines(
                K, width, height, far_plane, T_world_from_cam, scale=scale_current
            )
            self.current_lineset = o3d.geometry.LineSet()
            self.current_lineset.points = o3d.utility.Vector3dVector(points_curr)
            self.current_lineset.lines = o3d.utility.Vector2iVector(lines_curr)
            self.current_lineset.colors = o3d.utility.Vector3dVector([color for _ in lines_curr])

            self.vis.add_geometry(self.current_lineset, reset_bounding_box=False)

            # ========== 添加当前相机坐标系（调试方向）==========
            if show_coordinate_frames:
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                frame.transform(T_world_from_cam)
                self.current_frame = frame
                self.vis.add_geometry(self.current_frame, reset_bounding_box=False)

            # ========== 创建历史视锥体（小且淡）==========
            for offset in range(1, min(history_length, idx + 1)):
                j = idx - offset
                fade = 1.0 - 0.7 * (offset / history_length)
                T_cam_from_world_hist = self.extrinsics[j]
                T_cam_from_world_hist = T_cam_from_world_hist @ T_AC  # 应用校正变换
                T_world_from_cam_hist = T_cam_from_world_hist

                K_hist = self.intrinsics[j]
                color_hist = self.colors[j]


                points_hist, lines_hist = create_frustum_lines(
                    K_hist, width, height, far_plane, T_world_from_cam_hist, scale=1.0
                )
                lineset_hist = o3d.geometry.LineSet()
                lineset_hist.points = o3d.utility.Vector3dVector(points_hist)
                lineset_hist.lines = o3d.utility.Vector2iVector(lines_hist)
                faded_color = np.array(color_hist) * fade
                lineset_hist.colors = o3d.utility.Vector3dVector([faded_color for _ in lines_hist])

                self.vis.add_geometry(lineset_hist, reset_bounding_box=False)
                self.history_linesets.append(lineset_hist)

            # ========== 刷新画面 ==========
            self.vis.poll_events()
            self.vis.update_renderer()

            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)

        print("播放完成。可继续查看，按 ESC 退出。")
        while self.vis.poll_events():
            self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


# ================== 主函数 ==================
def main():
    # 检查文件是否存在
    for path in [ply_path, extrinsics_csv, intrinsics_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到文件: {path}")

    print("Loading 3D mesh...")
    mesh = read_ply(ply_path)

    print("Loading extrinsics and intrinsics...")
    _, extrinsics = read_extrinsics(extrinsics_csv)
    intrinsics = read_intrinsics(intrinsics_csv)

    assert len(extrinsics) == len(intrinsics), "外参与内参数量不匹配！"

    print(f"共 {len(extrinsics)} 帧相机位姿，开始可视化...")

    # 启动动画
    animator = AutoFrustumAnimator(mesh, extrinsics, intrinsics)
    try:
        animator.run()
    finally:
        animator.close()


if __name__ == "__main__":
    main()