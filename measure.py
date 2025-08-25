import numpy as np
import cv2
import pandas as pd
import os

# -------------------------------
# 🔧 原有函数（保持不变）
# -------------------------------
def read_camera_matrix(camera_matrix_path, target_size=(256, 192), original_size=(1920, 1080)):
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
    return {
        'fx': fx_new,
        'fy': fy_new,
        'cx': cx_new,
        'cy': cy_new,
        'width': target_w,
        'height': target_h
    }

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q[3], q[0], q[1], q[2]
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
    ])
    return R

def read_odometry_csv(file_path):
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

class VideoLoader:
    def __init__(self, video_path, target_size=(256, 192)):
        self.video_path = video_path
        self.target_size = target_size
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame_id = -1
        self._current_frame = None
        self.orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self, frame_idx, return_original=False):
        if frame_idx == self._current_frame_id:
            return self._current_frame_orig if return_original else self._current_frame
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._current_frame_orig = frame_rgb
        if not return_original:
            frame_resized = cv2.resize(frame_rgb, self.target_size, interpolation=cv2.INTER_AREA)
            self._current_frame = frame_resized
        else:
            self._current_frame = None
        self._current_frame_id = frame_idx
        return self._current_frame_orig if return_original else self._current_frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

def get_depth_image(depth_path):
    if not os.path.exists(depth_path):
        return None
    try:
        img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        return img / 1000.0  # 转为米
    except Exception as e:
        print(f"Error loading depth image {depth_path}: {e}")
        return None

def main():
    base_dir = '0018cc1438'
    depth_dir = os.path.join(base_dir, 'depth')
    odometry_file = os.path.join(base_dir, 'odometry.csv')
    rgb_video = os.path.join(base_dir, 'rgb.mp4')
    camera_matrix_path = os.path.join(base_dir, 'camera_matrix.csv')

    # ⚙️ 尺寸定义
    rgb_size = (1920, 1440)
    depth_size = (256, 192)  # camera matrix 和 depth map 的分辨率
    display_width = 960
    display_height = int(1440 * (display_width / 1920))

    # 坐标映射比例
    scale_display_to_rgb_x = rgb_size[0] / display_width
    scale_display_to_rgb_y = rgb_size[1] / display_height
    scale_rgb_to_depth_x = depth_size[0] / rgb_size[0]
    scale_rgb_to_depth_y = depth_size[1] / rgb_size[1]

    camera_params = read_camera_matrix(
        camera_matrix_path,
        target_size=depth_size,
        original_size=rgb_size
    )

    if not os.path.exists(odometry_file):
        print(f"❌ Odometry file not found: {odometry_file}")
        return
    odometry_data = read_odometry_csv(odometry_file)
    rgb_loader = VideoLoader(rgb_video, target_size=depth_size)
    total_frames = rgb_loader.frame_count

    # 全局变量
    world_points_3d = []  # 存储两个3D世界坐标点
    distance_3d = 0.0      # 存储两点间距离
    window_name = "3D Measurement"

    # -------------------------------
    # 🖱️ 鼠标点击事件
    # -------------------------------
    def click_event(event, x, y, flags, params):
        nonlocal world_points_3d, distance_3d
        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. 显示坐标 → RGB坐标
            x_rgb = int(x * scale_display_to_rgb_x)
            y_rgb = int(y * scale_display_to_rgb_y)
            # 2. RGB坐标 → 深度图坐标
            x_d = int(x_rgb * scale_rgb_to_depth_x)
            y_d = int(y_rgb * scale_rgb_to_depth_y)

            frame_idx = cv2.getTrackbarPos('Frame', window_name)
            depth_path = os.path.join(depth_dir, f"{frame_idx:06d}.png")
            current_depth_image = get_depth_image(depth_path)
            current_pose = odometry_data.get(frame_idx)

            if current_depth_image is None or current_pose is None:
                print("No depth or pose data available.")
                return
            if y_d >= current_depth_image.shape[0] or x_d >= current_depth_image.shape[1]:
                print("Depth coordinate out of bounds.")
                return

            z = current_depth_image[y_d, x_d]
            if z <= 0:
                print("Invalid depth at clicked point.")
                return

            fx, fy, cx, cy = camera_params['fx'], camera_params['fy'], camera_params['cx'], camera_params['cy']
            X = (x_d - cx) * z / fx
            Y = (y_d - cy) * z / fy
            pt_camera = np.array([X, Y, z])

            R = quaternion_to_rotation_matrix(current_pose['quaternion'])
            t = current_pose['position']
            pt_world = R @ pt_camera + t

            if len(world_points_3d) < 2:
                world_points_3d.append(pt_world)
                print(f"3D Point {len(world_points_3d)}: {pt_world}")
            else:
                world_points_3d[0] = world_points_3d[1]
                world_points_3d[1] = pt_world
                print(f"Updated 3D Points: {world_points_3d}")

            # 更新距离
            if len(world_points_3d) == 2:
                distance_3d = np.linalg.norm(world_points_3d[1] - world_points_3d[0])
                print(f"3D Distance: {distance_3d:.3f} meters")

    # -------------------------------
    # 🎬 主交互循环
    # -------------------------------
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.createTrackbar('Frame', window_name, 10, total_frames - 1, lambda x: None)
    cv2.setMouseCallback(window_name, click_event)

    print("拖动进度条切换帧")
    print("点击两个点定义3D空间线段")
    print("实时显示两点间距离")
    print("←/→ 快退/快进 | q: 退出")

    while True:
        frame_idx = cv2.getTrackbarPos('Frame', window_name)
        rgb_frame_orig = rgb_loader.get_frame(frame_idx, return_original=True)
        if rgb_frame_orig is None:
            continue

        # 缩放用于显示
        frame_display = cv2.resize(rgb_frame_orig, (display_width, display_height), interpolation=cv2.INTER_AREA)
        frame_display_bgr = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)

        # 获取当前位姿
        if frame_idx not in odometry_data:
            cv2.imshow(window_name, frame_display_bgr)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): break
            continue

        data = odometry_data[frame_idx]
        R_inv = quaternion_to_rotation_matrix(data['quaternion']).T
        t_inv = -R_inv @ data['position']
        fx, fy, cx, cy = camera_params['fx'], camera_params['fy'], camera_params['cx'], camera_params['cy']

        # 投影点并画线
        projected_points_2d = []
        if len(world_points_3d) > 0:
            colors = [(0, 255, 0), (255, 0, 0)]
            for i, pt_world in enumerate(world_points_3d):
                pt_camera = R_inv @ pt_world + t_inv
                if pt_camera[2] <= 0:
                    projected_points_2d.append(None)
                    continue
                u_d = int((pt_camera[0] * fx / pt_camera[2]) + cx)
                v_d = int((pt_camera[1] * fy / pt_camera[2]) + cy)
                u_rgb = int(u_d / scale_rgb_to_depth_x)
                v_rgb = int(v_d / scale_rgb_to_depth_y)
                u_disp = int(u_rgb * (display_width / rgb_size[0]))
                v_disp = int(v_rgb * (display_height / rgb_size[1]))
                if 0 <= u_disp < display_width and 0 <= v_disp < display_height:
                    projected_points_2d.append((u_disp, v_disp))
                    cv2.circle(frame_display_bgr, (u_disp, v_disp), 8, colors[i], -1)
                    cv2.putText(frame_display_bgr, f'P{i+1}', (u_disp + 15, v_disp),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
                else:
                    projected_points_2d.append(None)

            # ✅ 绘制连接线
            if len(projected_points_2d) == 2 and None not in projected_points_2d:
                cv2.line(frame_display_bgr, projected_points_2d[0], projected_points_2d[1], (255, 255, 0), 2, lineType=cv2.LINE_AA)
                # ✅ 显示距离文本（放在中点上方）
                mid_x = (projected_points_2d[0][0] + projected_points_2d[1][0]) // 2
                mid_y = (projected_points_2d[0][1] + projected_points_2d[1][1]) // 2 - 10
                cv2.putText(frame_display_bgr, f'{distance_3d:.4f}m', (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # 添加图例
        #cv2.putText(frame_display_bgr, "P1: Green", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        #cv2.putText(frame_display_bgr, "P2: Blue", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        #cv2.putText(frame_display_bgr, "Line: Yellow", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow(window_name, frame_display_bgr)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('→') or key == 83:
            new_pos = min(frame_idx + 1, total_frames - 1)
            cv2.setTrackbarPos('Frame', window_name, new_pos)
        elif key == ord('←') or key == 81:
            new_pos = max(frame_idx - 1, 0)
            cv2.setTrackbarPos('Frame', window_name, new_pos)

    rgb_loader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()