import numpy as np
import cv2
import pandas as pd
import open3d as o3d
import open3d.core as o3c
import os
import time

def save_colmap_compatible_ply(pcd, filename):
    """将 tensor 点云保存为 COLMAP 兼容的 ASCII PLY"""
    pcd_legacy = pcd.to_legacy()
    points = np.asarray(pcd_legacy.points)
    colors = np.asarray(pcd_legacy.colors)
    colors_uchar = np.clip(colors * 255, 0, 255).astype(np.ubyte)
    assert points.shape[1] == 3 and colors.shape[1] == 3, "Invalid point or color shape"
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for pt, col in zip(points, colors_uchar):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {col[0]} {col[1]} {col[2]}\n")
    print(f"COLMAP-compatible PLY saved to {filename}")

def read_camera_matrix(camera_matrix_path, target_size=(256, 192), original_size=(1920, 1440)):
    """读取并缩放相机内参"""
    matrix = pd.read_csv(camera_matrix_path, header=None).values
    assert matrix.shape == (3, 3), "Camera matrix must be 3x3"
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    fx_new = fx * target_w / orig_w
    fy_new = fy * target_h / orig_h
    cx_new = cx * target_w / orig_w
    cy_new = cy * target_h / orig_h
    return o3d.camera.PinholeCameraIntrinsic(
        width=target_w,
        height=target_h,
        fx=fx_new,
        fy=fy_new,
        cx=cx_new,
        cy=cy_new
    )

def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵"""
    w, x, y, z = q[3], q[0], q[1], q[2]
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
    ])
    return R

def read_odometry_csv(file_path):
    """读取位姿数据"""
    df = pd.read_csv(file_path)
    odometry_data = {}
    for _, row in df.iterrows():
        frame = int(row[' frame'])
        odometry_data[frame] = {
            'timestamp': row['timestamp'],
            'position': np.array([row[' x'], row[' y'], row[' z']]),
            'quaternion': np.array([row[' qx'], row[' qy'], row[' qz'], row[' qw']])
        }
    return odometry_data

class RGBVideoLoader:
    """高效加载 RGB 视频帧"""
    def __init__(self, video_path, target_size=(256, 192)):
        self.video_path = video_path
        self.target_size = target_size
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame_id = -1
        self._current_frame = None

    def get_frame(self, frame_idx):
        if frame_idx == self._current_frame_id:
            return self._current_frame
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.target_size, interpolation=cv2.INTER_AREA)
        self._current_frame_id = frame_idx
        self._current_frame = frame_resized
        return frame_resized

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

def get_depth_image(depth_path):
    """加载深度图（单位：米）"""
    if not os.path.exists(depth_path):
        return None
    try:
        img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        return img / 1000.0  # 假设单位是毫米 → 转为米
    except Exception as e:
        print(f"Error loading depth image {depth_path}: {e}")
        return None

def depth_to_point_cloud(depth_image, intrinsic):
    """深度图转点云（仅深度有效范围筛选）"""
    h, w = depth_image.shape
    K = intrinsic.intrinsic_matrix
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    valid_mask = (z > 0.1) & (z < 10.0)
    valid_mask = valid_mask.ravel()
    return points[valid_mask], valid_mask

def color_point_cloud(rgb_frame, valid_mask):
    """提取对应颜色"""
    h, w, _ = rgb_frame.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.ravel()[valid_mask]
    v = v.ravel()[valid_mask]
    colors = rgb_frame[v, u] / 255.0
    return colors

def create_camera_frustum(R, t, intrinsic, scale=0.1, color=[1, 0.5, 0]):
    """
    创建一个相机视锥体（金字塔形状）
    - R: 旋转矩阵 (3,3)
    - t: 平移向量 (3,)
    - intrinsic: PinholeCameraIntrinsic
    - scale: 控制视锥体大小（表示视锥体深度）
    - color: 颜色（默认橙色）
    """
    # 从内参矩阵提取参数
    K = intrinsic.intrinsic_matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    width, height = intrinsic.width, intrinsic.height

    # 计算在 depth = scale 处的四个角（归一化设备坐标）
    x_right = scale * (width - cx) / fx
    x_left = scale * (0 - cx) / fx
    y_bottom = scale * (height - cy) / fy
    y_top = scale * (0 - cy) / fy

    # 相机坐标系下的点（深度为 scale）
    corners_cam = np.array([
        [0, 0, 0],           # 相机中心
        [x_left, y_top, scale],    # 远平面左上
        [x_right, y_top, scale],   # 远平面右上
        [x_right, y_bottom, scale],# 远平面右下
        [x_left, y_bottom, scale]  # 远平面左下
    ])  # (5, 3)

    # 转世界坐标：P_world = R @ P_cam + t
    corners_world = (R @ corners_cam.T).T + t  # (5, 3)

    # 定义线段
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # 中心到四角
        [1, 2], [2, 3], [3, 4], [4, 1]   # 底部矩形
    ]

    # 创建 LineSet
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(corners_world)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

    return frustum

def main():
    base_dir = '0018cc1438'  # 替换为实际路径
    depth_dir = os.path.join(base_dir, 'depth')
    odometry_file = os.path.join(base_dir, 'odometry.csv')
    rgb_video = os.path.join(base_dir, 'mask.mp4')
    camera_matrix_path = os.path.join(base_dir, 'camera_matrix.csv')
    output_ply = 'results/fused_pointcloud_0018cc1438.ply'

    # -------------------------------
    # 🔧 参数设置
    # -------------------------------
    target_size = (256, 192)
    frame_step = 5
    voxel_size = 0.02
    original_rgb_size = (1920, 1440)
    vis_update_freq = 3  # 每几帧更新一次可视化

    # -------------------------------
    # 🚀 设备选择：CUDA > CPU
    # -------------------------------
    device = o3c.Device("CUDA:0") if o3c.cuda.is_available() else o3c.Device("CPU:0")
    print(f"Using device: {device}")

    # -------------------------------
    # 🚦 初始化
    # -------------------------------
    print("Loading camera matrix...")
    camera_intrinsics = read_camera_matrix(
        camera_matrix_path,
        target_size=target_size,
        original_size=original_rgb_size
    )

    print("Loading odometry...")
    if not os.path.exists(odometry_file):
        print(f"❌ Odometry file not found: {odometry_file}")
        return
    odometry_data = read_odometry_csv(odometry_file)

    global_pcd = o3d.t.geometry.PointCloud(device)

    # 预生成深度文件路径
    depth_files = {f: os.path.join(depth_dir, f"{f:06d}.png") for f in odometry_data.keys()}
    available_frames = sorted([
        f for f in odometry_data.keys()
        if os.path.exists(depth_files[f])
    ])
    sampled_frames = available_frames[::frame_step]
    if len(sampled_frames) == 0:
        print("❌ No valid frames found.")
        return
    print(f"Processing {len(sampled_frames)} frames at {target_size}")

    # -------------------------------
    # 🎥 视频加载器
    # -------------------------------
    rgb_loader = RGBVideoLoader(rgb_video, target_size=target_size)

    # -------------------------------
    # ✅ 可视化设置
    # -------------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Reconstruction + Camera Trajectory", width=1280, height=720)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2
    render_opt.background_color = np.array([0, 0, 0])
    first_update = True
    first_pcd_added = False
    legacy_display_pcd = o3d.geometry.PointCloud()
    frame_times = []

    # ✅ 存储所有有效相机位姿（用于最终绘制轨迹和视锥体）
    camera_positions = []
    all_frustums = []

    # -------------------------------
    # 🔁 主循环
    # -------------------------------
    for i, frame_idx in enumerate(sampled_frames):
        start_time = time.time()

        # 加载数据
        rgb_frame = rgb_loader.get_frame(frame_idx)
        depth_image = get_depth_image(depth_files[frame_idx])

        if rgb_frame is None or depth_image is None:
            print(f"⚠️  Missing data for frame {frame_idx}. Skipping.")
            continue

        # 实时显示当前 RGB 帧
        rgb_frame_rotated = cv2.rotate(rgb_frame, cv2.ROTATE_90_CLOCKWISE)
        display_size = (384, 512)
        rgb_frame_enlarged = cv2.resize(rgb_frame_rotated, display_size, interpolation=cv2.INTER_LINEAR)
        rgb_frame_bgr = cv2.cvtColor(rgb_frame_enlarged, cv2.COLOR_RGB2BGR)
        cv2.imshow("Live RGB", rgb_frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Real-time video display stopped by user.")
            break

        # 位姿
        data = odometry_data[frame_idx]
        R = quaternion_to_rotation_matrix(data['quaternion'])
        t = data['position']

        # 深度转点云
        points_camera, valid_mask = depth_to_point_cloud(depth_image, camera_intrinsics)
        if len(points_camera) == 0:
            print(f"⚠️  No valid points in depth image for frame {frame_idx}. Skipping.")
            continue

        # 转世界坐标
        points_world = (R @ points_camera.T).T + t

        # 提取颜色
        colors = color_point_cloud(rgb_frame, valid_mask)

        # 创建 GPU 点云
        pcd_frame = o3d.t.geometry.PointCloud(device)
        pcd_frame.point["positions"] = o3c.Tensor(points_world.astype(np.float32), device=device)
        pcd_frame.point["colors"] = o3c.Tensor(colors.astype(np.float32), device=device)
        pcd_frame = pcd_frame.voxel_down_sample(voxel_size=voxel_size)

        # 累积到全局点云
        if not first_pcd_added:
            global_pcd = pcd_frame
            first_pcd_added = True
            print("Initialized global point cloud.")
        else:
            global_pcd = global_pcd.voxel_down_sample(voxel_size=voxel_size * 2)
            global_pcd += pcd_frame

        # 🧹 显存/内存优化
        del points_camera, points_world, colors, pcd_frame, rgb_frame, depth_image, valid_mask

        # ✅ 保存有效位姿用于轨迹和视锥体
        camera_positions.append(t.copy())
        frustum = create_camera_frustum(R, t, camera_intrinsics, scale=0.1, color=[1, 0.5, 0])  # 橙色
        all_frustums.append(frustum)

        # -------------------------------
        # ✅ 可视化更新（仅点云）
        # -------------------------------
        if i % vis_update_freq == 0 or i == len(sampled_frames) - 1:
            num_global_points = len(global_pcd.point["positions"])
            if num_global_points > 50000:
                display_pcd = global_pcd.voxel_down_sample(voxel_size=voxel_size * 1.5)
            else:
                display_pcd = global_pcd

            legacy_display = display_pcd.to_legacy()

            if first_update:
                legacy_display_pcd.points = legacy_display.points
                legacy_display_pcd.colors = legacy_display.colors
                vis.add_geometry(legacy_display_pcd, reset_bounding_box=True)
                first_update = False
            else:
                legacy_display_pcd.points = legacy_display.points
                legacy_display_pcd.colors = legacy_display.colors
                vis.update_geometry(legacy_display_pcd)

        vis.poll_events()
        vis.update_renderer()
        end_time = time.time()
        frame_times.append(end_time - start_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        avg_fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0
        print(f"Frame {frame_idx:6d} | Points: {len(global_pcd.point['positions']):8d} | FPS: {avg_fps:4.1f}")

    # -------------------------------
    # 🧯 最终处理
    # -------------------------------
    if len(global_pcd.point["positions"]) == 0:
        print("❌ No points generated. Exiting.")
        vis.destroy_window()
        cv2.destroyAllWindows()
        rgb_loader.release()
        return

    print(f"\nFinal point cloud has {len(global_pcd.point['positions'])} points. Final down-sampling...")
    global_pcd = global_pcd.voxel_down_sample(voxel_size=voxel_size)
    #save_colmap_compatible_ply(global_pcd, output_ply)

    # -------------------------------
    # ✅ 最终可视化：添加轨迹和所有视锥体
    # -------------------------------
    vis.clear_geometries()

    # 添加全局点云
    legacy_final = global_pcd.to_legacy()
    vis.add_geometry(legacy_final)

    # 添加原点坐标系（红-X, 绿-Y, 蓝-Z）
    global_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(global_coord)

    # 添加所有相机视锥体
    for frustum in all_frustums:
        vis.add_geometry(frustum, reset_bounding_box=False)

    # 添加相机运动轨迹线
    if len(camera_positions) >= 2:
        lines = [[i, i+1] for i in range(len(camera_positions)-1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(camera_positions)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(lines))])  # 绿色
        vis.add_geometry(line_set, reset_bounding_box=False)

    vis.reset_view_point(True)
    print(f"\n✅ Done! Final points: {len(global_pcd.point['positions'])}")
    print("CloseOperation: Close window to exit.")
    vis.run()
    vis.destroy_window()

    # -------------------------------
    # 🧼 清理资源
    # -------------------------------
    cv2.destroyAllWindows()
    rgb_loader.release()

if __name__ == "__main__":
    main()