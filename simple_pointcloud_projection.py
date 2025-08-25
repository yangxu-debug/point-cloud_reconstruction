import os
import cv2
import numpy as np
import pandas as pd
from skimage.transform import resize
import argparse
import open3d as o3d
import json


# -------------------------------
# 工具函数
# -------------------------------

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


def get_correction_transform():
    """ARKit 坐标系校正：绕 X 轴旋转 180°"""
    T_AC = np.eye(4)
    T_AC[1, 1] = -1  # Y → -Y
    T_AC[2, 2] = -1  # Z → -Z
    return T_AC


def project_points_to_image_with_depth(vertices, intrinsic_matrix, T_world_to_camera, image_width, image_height, depth_map=None):
    """将3D点投影到2D图像平面，使用深度图进行遮挡检测，返回详细的可见性信息"""
    # 提取内参
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # 世界坐标到相机坐标的变换
    R = T_world_to_camera[:3, :3]
    t = T_world_to_camera[:3, 3]
    
    # 变换顶点到相机坐标系
    vertices_cam = (R @ vertices.T).T + t
    
    # 只保留相机前方的点
    valid_mask = vertices_cam[:, 2] > 0.1
    vertices_valid = vertices_cam[valid_mask]
    
    if len(vertices_valid) == 0:
        return [], [], valid_mask, [], []
    
    # 投影到图像平面
    u = (fx * vertices_valid[:, 0] / vertices_valid[:, 2] + cx).astype(int)
    v = (fy * vertices_valid[:, 1] / vertices_valid[:, 2] + cy).astype(int)
    
    # 裁剪到图像边界内
    inside_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    u_inside = u[inside_mask]
    v_inside = v[inside_mask]
    depths_inside = vertices_valid[inside_mask, 2]  # 相机坐标系下的深度
    
    # 获取原始顶点索引（用于跟踪哪个顶点对应哪个投影点）
    original_indices = np.where(valid_mask)[0][inside_mask]
    
    # 如果有深度图，进行遮挡检测
    if depth_map is not None:
        visible_mask = np.ones(len(u_inside), dtype=bool)
        
        for i, (u_coord, v_coord, point_depth) in enumerate(zip(u_inside, v_inside, depths_inside)):
            # 获取深度图中对应位置的深度值
            depth_from_map = depth_map[v_coord, u_coord]
            
            # 如果点云点的深度比深度图中的深度大（更远），则认为被遮挡
            # 添加一个小的容差值（比如0.1米）来处理浮点误差
            if point_depth > depth_from_map / 1000.0 + 0.1:  # 深度图是毫米，转换为米
                visible_mask[i] = False
        
        # 只保留可见的点
        u_visible = u_inside[visible_mask]
        v_visible = v_inside[visible_mask]
        visible_indices = original_indices[visible_mask]
        
        return u_visible, v_visible, valid_mask, visible_indices, depths_inside[visible_mask]
    
    return u_inside, v_inside, valid_mask, original_indices, depths_inside


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
        # Rotate 90° CCW and convert to RGB
        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()


# -------------------------------
# 主函数
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert custom dataset to ScanNet 2D format with textured mesh and debug projection.")
    parser.add_argument('--input_dir', default='20250820-212141', help='输入目录，包含 depth/, video.mov, intrinsics.csv, extrinsics.csv, mesh.ply')
    parser.add_argument('--output_path', default='output', help='输出根目录')
    parser.add_argument('--scene_name', default='scene0000_00', help='输出场景名称')
    parser.add_argument('--frame_skip', type=int, default=30, help='每隔几帧处理一次（1 表示全取）')
    parser.add_argument('--output_image_width', type=int, default=1920, help='输出图像宽度')
    parser.add_argument('--output_image_height', type=int, default=1440, help='输出图像高度')
    parser.add_argument('--original_rgb_width', type=int, default=1920, help='原始 RGB 图像宽度')
    parser.add_argument('--original_rgb_height', type=int, default=1440, help='原始 RGB 图像高度')
    parser.add_argument('--tolerance', type=float, default=0.033, help='时间对齐容忍度（秒）')
    parser.add_argument('--export_label_images', action='store_true', help='是否导出标签图像（本数据集通常没有）')
    parser.add_argument('--debug_projected_points', default=True,
                        help='调试模式：将 mesh 顶点投影到图像并保存可视化结果（用于检查内参/外参是否正确）')
    args = parser.parse_args()

    base_dir = args.input_dir
    output_dir = os.path.join(args.output_path, args.scene_name)
    os.makedirs(output_dir, exist_ok=True)

    # 创建输出子目录
    color_dir = os.path.join(output_dir, 'color')
    depth_dir = os.path.join(output_dir, 'depth')
    pose_dir = os.path.join(output_dir, 'pose')
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    if args.export_label_images:
        label_dir = os.path.join(output_dir, 'label')
        os.makedirs(label_dir, exist_ok=True)

  
    intrinsics_csv = os.path.join(base_dir, 'intrinsics_log.csv')
    extrinsics_csv = os.path.join(base_dir, 'extrinsics_log.csv')

    if not os.path.exists(intrinsics_csv):
        print(f"❌ Intrinsics file not found: {intrinsics_csv}")
        return
    if not os.path.exists(extrinsics_csv):
        print(f"❌ Extrinsics file not found: {extrinsics_csv}")
        return

    intrinsics_dict = read_csv_with_timestamp(intrinsics_csv)
    extrinsics_dict = read_csv_with_timestamp(extrinsics_csv)

    # -------------------------------
    # 3. 加载 RGB 视频
    # -------------------------------
    video_path = os.path.join(base_dir, 'video.mov')
    if not os.path.exists(video_path):
        print(f"❌ RGB video not found: {video_path}")
        return

    rgb_loader = RGBVideoLoader(video_path, fps=60)

    # -------------------------------
    # 4. 时间对齐并筛选有效帧
    # -------------------------------
    sorted_ts = sorted(extrinsics_dict.keys())
    valid_entries = []

    for idx, depth_ts in enumerate(sorted_ts):
        if idx % args.frame_skip != 0:
            continue

        closest_intrinsic_ts = find_closest_timestamp(depth_ts, list(intrinsics_dict.keys()), args.tolerance)
        closest_extrinsic_ts = find_closest_timestamp(depth_ts, list(extrinsics_dict.keys()), args.tolerance)

        if closest_intrinsic_ts is None or closest_extrinsic_ts is None:
            continue

        valid_entries.append({
            'depth_ts': depth_ts,
            'intrinsic_ts': closest_intrinsic_ts,
            'extrinsic_ts': closest_extrinsic_ts,
        })

    if not valid_entries:
        print("❌ No synchronized frames found.")
        return
    print(f"✅ {len(valid_entries)} frames will be exported.")

    # 设置目标图像尺寸
    target_size = (args.output_image_width, args.output_image_height)
    
    # 加载点云
    mesh_path = os.path.join(base_dir, 'mesh.ply')
    if not os.path.exists(mesh_path):
        print(f"❌ Mesh file not found: {mesh_path}")
        return
    
    print(f"🎨 Loading mesh from {mesh_path}...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0:
        print("⚠️ Empty mesh, skipping.")
        return
    
    vertices = np.asarray(mesh.vertices)
    print(f"✅ Loaded {len(vertices)} vertices")
    
    # 创建输出目录
    debug_dir = os.path.join(output_dir, 'debug_projected')
    os.makedirs(debug_dir, exist_ok=True)
    
    # 初始化顶点可见性跟踪数据结构
    vertex_visibility_data = {
        'mesh_info': {
            'total_vertices': len(vertices),
            'frame_count': len(valid_entries)
        },
        'vertices': {}  # 按顶点索引组织数据
    }
    
    # 初始化每个顶点的数据结构
    for i in range(len(vertices)):
        vertex_visibility_data['vertices'][str(i)] = {
            'vertex_index': i,
            'world_coords': vertices[i].tolist(),
            'visible_frames': []  # 记录在哪些帧中可见
        }
    
    # 初始化相机参数数据结构
    camera_params_data = {
        'mesh_info': {
            'total_vertices': len(vertices),
            'frame_count': len(valid_entries)
        },
        'frames': []  # 按帧组织相机参数
    }
    
    # 直接处理每一帧的点云投影
    frame_index = 0
    successful_projections = 0
    
    for entry in valid_entries:
        depth_ts = entry['depth_ts']
        print(f"\n处理第 {frame_index + 1}/{len(valid_entries)} 帧 (时间戳: {depth_ts:.3f})")
        
        # 1. 获取RGB帧
        rgb_frame = rgb_loader.get_frame_by_timestamp(depth_ts)
        if rgb_frame is None:
            print(f"  ⚠️ 无法获取RGB帧，跳过")
            frame_index += 1
            continue
        
        # 调整RGB帧尺寸
        rgb_resized = cv2.resize(rgb_frame, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 2. 获取相机参数
        intrinsic_row = intrinsics_dict[entry['intrinsic_ts']]
        extrinsic_row = extrinsics_dict[entry['extrinsic_ts']]
        
        # 构建内参矩阵
        intrinsic_matrix = np.array([
            [intrinsic_row['m00'], 0, intrinsic_row['m02']],
            [0, intrinsic_row['m11'], intrinsic_row['m12']],
            [0, 0, 1]
        ])
        
        # 构建外参矩阵
        extrinsic_matrix = np.array([
            [extrinsic_row['m00'], extrinsic_row['m01'], extrinsic_row['m02'], extrinsic_row['m03']],
            [extrinsic_row['m10'], extrinsic_row['m11'], extrinsic_row['m12'], extrinsic_row['m13']],
            [extrinsic_row['m20'], extrinsic_row['m21'], extrinsic_row['m22'], extrinsic_row['m23']],
            [extrinsic_row['m30'], extrinsic_row['m31'], extrinsic_row['m32'], extrinsic_row['m33']]
        ])
        
        # 应用ARKit校正
        T_AC = get_correction_transform()
        extrinsic_matrix = extrinsic_matrix @ T_AC
        T_world_to_camera = np.linalg.inv(extrinsic_matrix)
        
        # 3. 加载深度图（如果存在）
        depth_file_pattern = f"{depth_ts:.6f}_W256_H192_DepthFloat32.raw"
        depth_dir_path = os.path.join(base_dir, 'depth')
        depth_file_path = os.path.join(depth_dir_path, depth_file_pattern)
        
        depth_map = None
        if os.path.exists(depth_file_path):
            try:
                # 加载深度图
                depth_data = np.fromfile(depth_file_path, dtype=np.float32)
                depth_img = depth_data.reshape((192, 256))
                
                # 调整深度图尺寸以匹配输出图像
                depth_resized = resize(depth_img, target_size[::-1], order=0, preserve_range=True, anti_aliasing=False)
                
                # 转换为毫米单位
                depth_map = (depth_resized * 1000).astype(np.float32)
                print(f"  ✅ 深度图已加载: {depth_map.shape}")
            except Exception as e:
                print(f"  ⚠️ 深度图加载失败: {e}")
        else:
            print(f"  ⚠️ 未找到深度图: {depth_file_pattern}")
        
        # 4. 执行点云投影
        print(f"  🔍 执行点云投影...")
        u_proj, v_proj, valid_mask, visible_indices, depths = project_points_to_image_with_depth(
            vertices, intrinsic_matrix, T_world_to_camera, 
            target_size[0], target_size[1], depth_map
        )
        
        if len(u_proj) > 0:
            # 记录每个可见顶点的信息
            for i, (u, v, vertex_idx, depth) in enumerate(zip(u_proj, v_proj, visible_indices, depths)):
                # 将可见性信息添加到对应顶点的记录中
                vertex_key = str(vertex_idx)
                frame_info = {
                    'frame_index': frame_index,
                    'timestamp': depth_ts,
                    'image_coords': [int(u), int(v)],
                    'depth': float(depth),
                    'intrinsic_matrix': intrinsic_matrix.tolist()  # 保存内参矩阵
                }
                vertex_visibility_data['vertices'][vertex_key]['visible_frames'].append(frame_info)
            
            # 5. 可视化投影结果
            rgb_debug = rgb_resized.copy()
            
            # 在图像上画投影点
            for u, v in zip(u_proj, v_proj):
                cv2.circle(rgb_debug, (u, v), radius=1, color=(255, 0, 0), thickness=-1)
            
            # 保存结果
            output_path = os.path.join(debug_dir, f"frame_{frame_index:04d}_ts_{depth_ts:.3f}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(rgb_debug, cv2.COLOR_RGB2BGR))
            
            print(f"  ✅ 投影成功: {len(u_proj)} 个点 -> {output_path}")
            successful_projections += 1
        else:
            print(f"  ⚠️ 没有有效投影点")
        
        # 将这一帧的相机参数添加到相机参数数据中
        frame_camera_info = {
            'frame_index': frame_index,
            'timestamp': depth_ts,
            'intrinsic_matrix': intrinsic_matrix.tolist(),
            'T_world_to_camera': T_world_to_camera.tolist()
        }
        camera_params_data['frames'].append(frame_camera_info)
        
        frame_index += 1
    
    print(f"\n=== 处理完成 ===")
    print(f"   总帧数: {len(valid_entries)}")
    print(f"   成功投影: {successful_projections} 帧")
    print(f"   结果保存在: {debug_dir}/")
    
    # 保存顶点可见性数据到JSON文件
    json_output_path = os.path.join(output_dir, 'vertex_visibility_data.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(vertex_visibility_data, f, indent=2, ensure_ascii=False)
    
    print(f"   顶点可见性数据保存到: {json_output_path}")
    
    # 保存相机参数数据到JSON文件
    camera_params_path = os.path.join(output_dir, 'camera_params_data.json')
    with open(camera_params_path, 'w', encoding='utf-8') as f:
        json.dump(camera_params_data, f, indent=2, ensure_ascii=False)
    
    print(f"   相机参数数据保存到: {camera_params_path}")
    
    # 统计信息
    total_visible_vertices = sum(len(vertex_data['visible_frames']) for vertex_data in vertex_visibility_data['vertices'].values())
    print(f"   总可见顶点次数: {total_visible_vertices}")
    
    # 计算每个顶点的可见性统计
    vertex_visibility_count = {}
    for vertex_key, vertex_data in vertex_visibility_data['vertices'].items():
        frame_count = len(vertex_data['visible_frames'])
        if frame_count > 0:
            vertex_visibility_count[vertex_key] = frame_count
    
    if vertex_visibility_count:
        max_visibility = max(vertex_visibility_count.values())
        min_visibility = min(vertex_visibility_count.values())
        avg_visibility = sum(vertex_visibility_count.values()) / len(vertex_visibility_count)
        print(f"   顶点可见性统计:")
        print(f"     最多可见帧数: {max_visibility}")
        print(f"     最少可见帧数: {min_visibility}")
        print(f"     平均可见帧数: {avg_visibility:.1f}")
        print(f"     有可见记录的顶点数: {len(vertex_visibility_count)}")
        
        # 显示一些示例顶点的可见性信息
        print(f"\n   示例顶点可见性信息:")
        sample_vertices = list(vertex_visibility_count.keys())[:3]  # 显示前3个顶点
        for vertex_key in sample_vertices:
            vertex_data = vertex_visibility_data['vertices'][vertex_key]
            print(f"     顶点 {vertex_key}: 在 {len(vertex_data['visible_frames'])} 帧中可见")
            if vertex_data['visible_frames']:
                first_frame = vertex_data['visible_frames'][0]
                print(f"       示例: 帧 {first_frame['frame_index']}, 坐标 {first_frame['image_coords']}")

    # -------------------------------
    # 清理资源
    # -------------------------------
    rgb_loader.release()
    print("✅ All done!")


if __name__ == "__main__":
    main()