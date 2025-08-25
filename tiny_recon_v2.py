import numpy as np
import cv2
import pandas as pd
import open3d as o3d
import open3d.core as o3c
import os
import time
import re
from scipy.spatial.transform import Rotation as R

def save_colmap_compatible_ply(pcd, filename):
    """保存点云为 COLMAP 兼容的 PLY 格式"""
    if pcd.is_empty():
        print(f"⚠️  Cannot save empty point cloud to {filename}")
        return
    pcd_legacy = pcd.to_legacy()
    points = np.asarray(pcd_legacy.points)
    colors = np.asarray(pcd_legacy.colors)
    colors_uchar = np.clip(colors * 255, 0, 255).astype(np.ubyte)
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


def read_csv_with_timestamp(csv_path):
    """读取 CSV，返回 timestamp -> row 的字典"""
    df = pd.read_csv(csv_path)
    data = {}
    for _, row in df.iterrows():
        ts = float(row['timestamp'])
        data[ts] = row
    return data


def find_closest_timestamp(target_ts, ts_list, tolerance=0.033):
    """在 ts_list 中找最接近 target_ts 的时间戳"""
    closest = min(ts_list, key=lambda ts: abs(ts - target_ts))
    if abs(closest - target_ts) < tolerance:
        return closest
    return None


def get_depth_image(depth_info):
    """
    加载 .raw 深度图
    depth_info: 字典，包含 'path', 'width', 'height'
    """
    path = depth_info['path']
    width = depth_info['width']
    height = depth_info['height']

    if not os.path.exists(path):
        return None

    try:
        # 读取 raw float32 数据
        data = np.fromfile(path, dtype=np.float32)
        if data.size != width * height:
            print(f"⚠️ Size mismatch: expected {width * height}, got {data.size} in {path}")
            return None

        # 重塑为图像
        img = data.reshape((height, width))

        # 旋转 90 度逆时针
        #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 假设单位是米；若为毫米，请取消下一行注释
        # img = img / 1000.0

        return img.astype(np.float32)
    except Exception as e:
        print(f"Error loading depth image {path}: {e}")
        return None


def depth_to_point_cloud(depth_image, intrinsic):
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
    h, w, _ = rgb_frame.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.ravel()[valid_mask]
    v = v.ravel()[valid_mask]
    colors = rgb_frame[v, u] / 255.0
    return colors


def create_camera_frustum(R, t, intrinsic, scale=0.3, color=[1, 0.5, 0]):
    K = intrinsic.intrinsic_matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    width, height = intrinsic.width, intrinsic.height

    x_right = scale * (width - cx) / fx
    x_left = scale * (0 - cx) / fx
    y_bottom = scale * (height - cy) / fy
    y_top = scale * (0 - cy) / fy

    corners_cam = np.array([
        [0, 0, 0],
        [x_left, y_top, scale],
        [x_right, y_top, scale],
        [x_right, y_bottom, scale],
        [x_left, y_bottom, scale]
    ])

    corners_world = (R @ corners_cam.T).T + t

    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(corners_world)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return frustum


class RGBVideoLoader:
    def __init__(self, video_path, fps=60):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        self.fps = fps
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video loaded: {self.frame_count} frames at {fps} FPS")

    def get_frame_by_timestamp(self, timestamp):
        frame_idx = int(round(timestamp * self.fps))
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        if self.cap.isOpened():
            self.cap.release()


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


def main():
    base_dir = '20250820-212141'
    depth_dir = os.path.join(base_dir, 'depth')
    extrinsics_file = os.path.join(base_dir, 'extrinsics_log.csv')
    intrinsics_file = os.path.join(base_dir, 'intrinsics_log.csv')
    rgb_video = os.path.join(base_dir, 'video.mov')
    output_ply = 'results/fused_pointcloud_aligned.ply'

    # -------------------------------
    # 🔧 参数
    # -------------------------------
    target_size = (256, 192)
    frame_step = 60
    voxel_size = 0.02
    original_rgb_size = (1920, 1440)
    vis_update_freq = 3
    fps = 60
    tolerance = 0.033  # ~1/60s

    # -------------------------------
    # 🚀 设备
    # -------------------------------
    device = o3c.Device("CUDA:0") if o3c.cuda.is_available() else o3c.Device("CPU:0")
    print(f"Using device: {device}")

    # -------------------------------
    # 📂 1. 获取所有深度图的时间戳和尺寸（.raw）
    # -------------------------------
    depth_files = {}
    for f in os.listdir(depth_dir):
        match = re.match(r'^(\d+\.?\d*)_W(\d+)_H(\d+)_DepthFloat32\.raw$', f)
        if match:
            ts = float(match.group(1))
            width = int(match.group(2))
            height = int(match.group(3))
            depth_files[ts] = {
                'path': os.path.join(depth_dir, f),
                'width': width,
                'height': height
            }
    if not depth_files:
        print("❌ No valid .raw depth files found.")
        return
    print(f"Found {len(depth_files)} depth .raw files")

    # -------------------------------
    # 📄 2. 加载内参与外参
    # -------------------------------
    if not os.path.exists(intrinsics_file):
        print(f"❌ Intrinsics file not found: {intrinsics_file}")
        return
    intrinsics_dict = read_csv_with_timestamp(intrinsics_file)

    if not os.path.exists(extrinsics_file):
        print(f"❌ Extrinsics file not found: {extrinsics_file}")
        return
    extrinsics_dict = read_csv_with_timestamp(extrinsics_file)

    # -------------------------------
    # 🎥 3. 加载视频
    # -------------------------------
    if not os.path.exists(rgb_video):
        print(f"❌ RGB video not found: {rgb_video}")
        return
    rgb_loader = RGBVideoLoader(rgb_video, fps=fps)

    # -------------------------------
    # 🔍 4. 时间戳对齐
    # -------------------------------
    valid_entries = []
    for depth_ts in sorted(depth_files.keys()):
        if int(round(depth_ts * fps)) % frame_step != 0:
            continue

        closest_intrinsic_ts = find_closest_timestamp(depth_ts, list(intrinsics_dict.keys()), tolerance)
        closest_extrinsic_ts = find_closest_timestamp(depth_ts, list(extrinsics_dict.keys()), tolerance)

        if closest_intrinsic_ts is None or closest_extrinsic_ts is None:
            continue

        valid_entries.append({
            'depth_ts': depth_ts,
            'intrinsic_ts': closest_intrinsic_ts,
            'extrinsic_ts': closest_extrinsic_ts,
            'depth_path': depth_files[depth_ts]
        })

    if not valid_entries:
        print("❌ No synchronized frames found.")
        return
    print(f"✅ {len(valid_entries)} synchronized frames will be processed.")

    # -------------------------------
    # ✅ 初始化
    # -------------------------------
    global_pcd = o3d.t.geometry.PointCloud(device)
    camera_positions = []
    all_frustums = []

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Reconstruction", width=1280, height=720)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2
    render_opt.background_color = np.array([0, 0, 0])
    first_update = True
    legacy_display_pcd = o3d.geometry.PointCloud()
    frame_times = []

    # -------------------------------
    # 🔁 主循环
    # -------------------------------
    for i, entry in enumerate(valid_entries):
        start_time = time.time()
        depth_ts = entry['depth_ts']

        # 加载深度图
        depth_image = get_depth_image(entry['depth_path'])
        if depth_image is None:
            continue

        # 加载内参
        row_in = intrinsics_dict[entry['intrinsic_ts']]
        K = np.array([[row_in[f'm{i}{j}'] for j in range(3)] for i in range(3)])
        orig_w, orig_h = original_rgb_size
        target_w, target_h = target_size
        K[0,0] *= target_w / orig_w
        K[1,1] *= target_h / orig_h
        K[0,2] *= target_w / orig_w
        K[1,2] *= target_h / orig_h
        intrinsic = o3d.camera.PinholeCameraIntrinsic(target_w, target_h, K[0,0], K[1,1], K[0,2], K[1,2])

        # 加载外参并校正坐标系
        row_ex = extrinsics_dict[entry['extrinsic_ts']]
        T_world_to_camera_original = np.array([[row_ex[f'm{i}{j}'] for j in range(4)] for i in range(4)])
        T_AC = get_correction_transform()
        T_world_to_camera_corrected = T_world_to_camera_original @ T_AC
        T_camera_to_world = T_world_to_camera_corrected
        R_world = T_camera_to_world[:3, :3].astype(np.float32)
        t_world = T_camera_to_world[:3, 3].astype(np.float32)

        # 加载 RGB
        rgb_frame = rgb_loader.get_frame_by_timestamp(depth_ts)
        if rgb_frame is None:
            continue
        rgb_frame = cv2.resize(rgb_frame, target_size, interpolation=cv2.INTER_AREA)

        # 转点云
        points_camera, valid_mask = depth_to_point_cloud(depth_image, intrinsic)
        if len(points_camera) == 0:
            continue

        points_world = (R_world @ points_camera.T).T + t_world
        colors = color_point_cloud(rgb_frame, valid_mask)

        pcd_frame = o3d.t.geometry.PointCloud(device)
        pcd_frame.point["positions"] = o3c.Tensor(points_world.astype(np.float32), device=device)
        pcd_frame.point["colors"] = o3c.Tensor(colors.astype(np.float32), device=device)
        #pcd_frame = pcd_frame.voxel_down_sample(voxel_size=voxel_size)

        if pcd_frame.is_empty():
            continue

        # 累积点云
        if global_pcd.is_empty():
            global_pcd = pcd_frame
        else:
            #global_pcd = global_pcd.voxel_down_sample(voxel_size=voxel_size * 2)
            global_pcd += pcd_frame

        # 记录相机位置和视锥
        camera_positions.append(t_world.copy())
        frustum = create_camera_frustum(R_world, t_world, intrinsic, scale=0.05, color=[1, 0.5, 0])
        all_frustums.append(frustum)

        # 显示 RGB
        rgb_display = cv2.resize(rgb_frame, (256, 192))
        rgb_bgr = cv2.rotate(cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR), cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("Live RGB", rgb_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 可视化更新
        if i % vis_update_freq == 0 or i == len(valid_entries) - 1:
            if not global_pcd.is_empty():
                num_pts = len(global_pcd.point["positions"])
                #display_pcd = global_pcd.voxel_down_sample(voxel_size=voxel_size * 1.5) if num_pts > 100000 else global_pcd
                display_pcd = global_pcd
                legacy_display = display_pcd.to_legacy()
                if first_update:
                    legacy_display_pcd.points = legacy_display.points
                    legacy_display_pcd.colors = legacy_display.colors
                    vis.add_geometry(legacy_display_pcd)
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
        print(f"Ts {depth_ts:.3f} | Points: {len(global_pcd.point['positions']) if not global_pcd.is_empty() else 0:8d} | FPS: {avg_fps:4.1f}")

    # -------------------------------
    # 🧯 最终处理：保存 + 可视化（含 mesh.ply）
    # -------------------------------
    if global_pcd.is_empty():
        print("❌ No points generated.")
    else:
        global_pcd = global_pcd.voxel_down_sample(voxel_size=voxel_size)
        os.makedirs("results", exist_ok=True)
        save_colmap_compatible_ply(global_pcd, output_ply)

        # 清空并重新添加所有几何体
        vis.clear_geometries()

        # 1. 添加重建点云
        vis.add_geometry(global_pcd.to_legacy())

        # 2. 添加坐标系原点
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0]))

        # 3. 添加相机轨迹
        if len(camera_positions) >= 2:
            lines = [[i, i+1] for i in range(len(camera_positions)-1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(camera_positions)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(lines))])
            vis.add_geometry(line_set)

        # 4. 添加相机视锥
        for frustum in all_frustums:
            vis.add_geometry(frustum, reset_bounding_box=False)

        # 5. 加载并添加 mesh.ply（半透明）
        mesh_path = os.path.join(os.path.dirname(depth_dir), 'mesh.ply')
        if os.path.exists(mesh_path):
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if mesh.is_empty():
                print(f"⚠️  Loaded mesh is empty: {mesh_path}")
            else:
                print(f"✅ Loaded mesh: {mesh_path} | {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                if not mesh.has_vertex_normals():
                    mesh.compute_vertex_normals()        
                vis.add_geometry(mesh, reset_bounding_box=False)      
        else:
            print(f"🟡 Mesh file not found: {mesh_path}")

        # 重置视角
        vis.reset_view_point(True)
        print(f"\n✅ Reconstruction complete! Final point count: {len(global_pcd.point['positions'])}")
        print("🔍 You can now inspect alignment between point cloud and mesh (semi-transparent).")
        vis.run()
        vis.destroy_window()

    # 清理资源
    cv2.destroyAllWindows()
    rgb_loader.release()


if __name__ == "__main__":
    main()