import numpy as np
import cv2
import pandas as pd
import os
import re
from scipy.spatial.transform import Rotation as R

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

def get_depth_image_raw(depth_path, width, height):
    """加载 .raw 深度图（float32）"""
    if not os.path.exists(depth_path):
        return None
    try:
        data = np.fromfile(depth_path, dtype=np.float32)
        if data.size != width * height:
            print(f"⚠️ Size mismatch: expected {width * height}, got {data.size}")
            return None
        img = data.reshape((height, width))
        return img  # 单位：米（假设已为米）
    except Exception as e:
        print(f"Error loading depth image {depth_path}: {e}")
        return None

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q[3], q[0], q[1], q[2]
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
    ])
    return R

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
        # 旋转 90°（根据需要调整）
        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

# -------------------------------
# 🧩 数据预加载（主入口）
# -------------------------------
def main():
    base_dir = '20250820-212141'  # 修改为你自己的路径
    depth_dir = os.path.join(base_dir, 'depth')
    extrinsics_file = os.path.join(base_dir, 'extrinsics_log.csv')
    intrinsics_file = os.path.join(base_dir, 'intrinsics_log.csv')
    rgb_video = os.path.join(base_dir, 'video.mov')

    # ⚙️ 参数
    target_size = (256, 192)
    original_rgb_size = (1920, 1440)
    fps = 60
    tolerance = 0.033
    display_width = 960
    display_height = int(1440 * (display_width / 1920))

    T_AC = get_correction_transform()
    # 坐标映射比例
    scale_display_to_rgb_x = original_rgb_size[0] / display_width
    scale_display_to_rgb_y = original_rgb_size[1] / display_height
    scale_rgb_to_depth_x = target_size[0] / original_rgb_size[0]
    scale_rgb_to_depth_y = target_size[1] / original_rgb_size[1]

    # -------------------------------
    # 📂 1. 加载所有 .raw 深度图（时间戳 → 路径）
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
    intrinsics_dict = read_csv_with_timestamp(intrinsics_file)
    extrinsics_dict = read_csv_with_timestamp(extrinsics_file)

    # -------------------------------
    # 🎥 3. 加载视频
    # -------------------------------
    rgb_loader = RGBVideoLoader(rgb_video, fps=fps)

    # -------------------------------
    # 🔍 4. 时间戳对齐 → 构建同步帧列表
    # -------------------------------
    synced_frames = []
    for depth_ts in sorted(depth_files.keys()):
        closest_intrinsic_ts = find_closest_timestamp(depth_ts, list(intrinsics_dict.keys()), tolerance)
        closest_extrinsic_ts = find_closest_timestamp(depth_ts, list(extrinsics_dict.keys()), tolerance)
        if closest_intrinsic_ts and closest_extrinsic_ts:
            synced_frames.append({
                'timestamp': depth_ts,
                'depth_info': depth_files[depth_ts],
                'intrinsic_row': intrinsics_dict[closest_intrinsic_ts],
                'extrinsic_row': extrinsics_dict[closest_extrinsic_ts]
            })

    if not synced_frames:
        print("❌ No synchronized frames found.")
        return
    print(f"✅ {len(synced_frames)} synchronized frames available.")

    # -------------------------------
    # 🖱️ 鼠标事件 & 主窗口
    # -------------------------------
    world_points_3d = []
    distance_3d = 0.0
    window_name = "3D Measurement"
    current_frame_idx = 0  # 当前显示帧索引（在 synced_frames 中）

    def click_event(event, x, y, flags, params):
        nonlocal world_points_3d, distance_3d
        if event == cv2.EVENT_LBUTTONDOWN:
            # 显示坐标 → RGB坐标
            x_rgb = int(x * scale_display_to_rgb_x)
            y_rgb = int(y * scale_display_to_rgb_y)
            # RGB坐标 → 深度图坐标
            x_d = int(x_rgb * scale_rgb_to_depth_x)
            y_d = int(y_rgb * scale_rgb_to_depth_y)

            frame_data = synced_frames[current_frame_idx]
            depth_img = get_depth_image_raw(
                frame_data['depth_info']['path'],
                frame_data['depth_info']['width'],
                frame_data['depth_info']['height']
            )
            if depth_img is None or y_d >= depth_img.shape[0] or x_d >= depth_img.shape[1]:
                print("Invalid depth data.")
                return

            z = depth_img[y_d, x_d]
            if z <= 0:
                print("Invalid depth value.")
                return

            # 重建内参矩阵（缩放到 target_size）
            K = np.array([[frame_data['intrinsic_row'][f'm{i}{j}'] for j in range(3)] for i in range(3)])
            orig_w, orig_h = original_rgb_size
            target_w, target_h = target_size
            fx = K[0,0] * target_w / orig_w
            fy = K[1,1] * target_h / orig_h
            cx = K[0,2] * target_w / orig_w
            cy = K[1,2] * target_h / orig_h

            X = (x_d - cx) * z / fx
            Y = (y_d - cy) * z / fy
            pt_camera = np.array([X, Y, z])

            # 外参：T_world_to_camera
            T_world_to_camera = np.array([[frame_data['extrinsic_row'][f'm{i}{j}'] for j in range(4)] for i in range(4)])
            T_world_to_camera = T_world_to_camera @ T_AC
            R = T_world_to_camera[:3, :3]
            t = T_world_to_camera[:3, 3]
            pt_world = R @ pt_camera + t

            if len(world_points_3d) < 2:
                world_points_3d.append(pt_world)
                print(f"3D Point {len(world_points_3d)}: {pt_world}")
            else:
                world_points_3d[0] = world_points_3d[1]
                world_points_3d[1] = pt_world

            if len(world_points_3d) == 2:
                distance_3d = np.linalg.norm(world_points_3d[1] - world_points_3d[0])
                print(f"3D Distance: {distance_3d:.3f} meters")

    # -------------------------------
    # 🎬 主循环
    # -------------------------------
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.createTrackbar('Frame', window_name, 0, len(synced_frames) - 1, lambda x: None)
    cv2.setMouseCallback(window_name, click_event)

    print("拖动进度条切换帧")
    print("点击两个点定义3D空间线段")
    print("实时显示两点间距离")
    print("←/→ 快退/快进 | q: 退出")

    while True:
        current_frame_idx = cv2.getTrackbarPos('Frame', window_name)
        if current_frame_idx >= len(synced_frames):
            continue
        frame_data = synced_frames[current_frame_idx]

        # 读取 RGB 帧
        rgb_frame = rgb_loader.get_frame_by_timestamp(frame_data['timestamp'])
        if rgb_frame is None:
            continue
        rgb_resized = cv2.resize(rgb_frame, target_size, interpolation=cv2.INTER_AREA)
        frame_display = cv2.resize(rgb_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
        frame_display_bgr = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)

        # 获取相机参数
        K = np.array([[frame_data['intrinsic_row'][f'm{i}{j}'] for j in range(3)] for i in range(3)])
        orig_w, orig_h = original_rgb_size
        target_w, target_h = target_size
        fx = K[0,0] * target_w / orig_w
        fy = K[1,1] * target_h / orig_h
        cx = K[0,2] * target_w / orig_w
        cy = K[1,2] * target_h / orig_h


        T_world_to_camera = np.array([[frame_data['extrinsic_row'][f'm{i}{j}'] for j in range(4)] for i in range(4)])
        T_world_to_camera = T_world_to_camera @ T_AC
        R_world_to_camera = T_world_to_camera[:3, :3]
        t_world_to_camera = T_world_to_camera[:3, 3]

        # 反投影：世界点 → 图像
        projected_points_2d = []
        if len(world_points_3d) > 0:
            colors = [(0, 255, 0), (255, 0, 0)]
            for i, pt_world in enumerate(world_points_3d):
                pt_camera = R_world_to_camera.T @ (pt_world - t_world_to_camera)
                if pt_camera[2] <= 0:
                    projected_points_2d.append(None)
                    continue
                u_d = int((pt_camera[0] * fx / pt_camera[2]) + cx)
                v_d = int((pt_camera[1] * fy / pt_camera[2]) + cy)
                u_rgb = int(u_d / scale_rgb_to_depth_x)
                v_rgb = int(v_d / scale_rgb_to_depth_y)
                u_disp = int(u_rgb * (display_width / original_rgb_size[0]))
                v_disp = int(v_rgb * (display_height / original_rgb_size[1]))
                if 0 <= u_disp < display_width and 0 <= v_disp < display_height:
                    projected_points_2d.append((u_disp, v_disp))
                    cv2.circle(frame_display_bgr, (u_disp, v_disp), 8, colors[i], -1)
                    cv2.putText(frame_display_bgr, f'P{i+1}', (u_disp + 15, v_disp),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
                else:
                    projected_points_2d.append(None)

            # 画线和距离
            if len(projected_points_2d) == 2 and None not in projected_points_2d:
                cv2.line(frame_display_bgr, projected_points_2d[0], projected_points_2d[1], (255, 255, 0), 2)
                mid_x = (projected_points_2d[0][0] + projected_points_2d[1][0]) // 2
                mid_y = (projected_points_2d[0][1] + projected_points_2d[1][1]) // 2 - 10
                cv2.putText(frame_display_bgr, f'{distance_3d:.4f}m', (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(window_name, frame_display_bgr)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('→') or key == 83:
            new_pos = min(current_frame_idx + 1, len(synced_frames) - 1)
            cv2.setTrackbarPos('Frame', window_name, new_pos)
        elif key == ord('←') or key == 81:
            new_pos = max(current_frame_idx - 1, 0)
            cv2.setTrackbarPos('Frame', window_name, new_pos)

    rgb_loader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()