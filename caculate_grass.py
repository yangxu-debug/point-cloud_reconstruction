import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import ConvexHull

try:
    import alphashape
except ImportError:
    raise ImportError("Please install: pip install alphashape")

try:
    from simplification.cutil import simplify_coords
except ImportError:
    raise ImportError("Please install: pip install simplification")


def downsample_2d(points_2d, factor=0.1):
    if len(points_2d) == 0:
        return points_2d
    bins = np.floor(points_2d / factor)
    _, idx = np.unique(bins, axis=0, return_index=True)
    return points_2d[idx]


def smooth_boundary(coords, tolerance=0.05):
    if len(coords) <= 3:
        return coords
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    simplified = simplify_coords(coords, tolerance)
    if len(simplified) > 1 and np.allclose(simplified[0], simplified[-1]):
        simplified = simplified[:-1]
    return simplified


def create_thick_boundary_line(boundary_3d, radius=0.05, color=[0, 1, 0]):
    mesh = o3d.geometry.TriangleMesh()
    n = len(boundary_3d)

    for i in range(n):
        p0 = boundary_3d[i]
        p1 = boundary_3d[(i + 1) % n]
        center = (p0 + p1) / 2.0
        direction = p1 - p0
        length = np.linalg.norm(direction)
        if length == 0:
            continue

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
        cylinder.compute_vertex_normals()

        z_axis = np.array([0, 0, 1])
        axis = direction / length
        c = np.dot(z_axis, axis)
        if abs(c) > 1.0:
            continue
        if c > 0.999:
            R = np.eye(3)
        elif c < -0.999:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi)
        else:
            rot_axis = np.cross(z_axis, axis)
            rot_axis /= np.linalg.norm(rot_axis)
            angle = np.arccos(c)
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
        cylinder.rotate(R, center=np.zeros(3))
        cylinder.translate(center)

        cylinder.paint_uniform_color(color)
        mesh += cylinder

    return mesh


def analyze_grass_area(semantic_ply_path, rgb_ply_path, grass_color=(4, 250, 7),
                       color_tol=0.1, y_tolerance=0.3, alpha=1.5,
                       downsample_2d_factor=0.1, simplify_tolerance=0.05,
                       boundary_radius=0.05, boundary_color=[0, 1, 0],
                       show_2d=True, show_3d=True, save_dir=None):
    print("Loading semantic point cloud (for analysis)...")
    pcd_semantic = o3d.io.read_point_cloud(semantic_ply_path)
    if not pcd_semantic.has_colors():
        raise ValueError("Semantic point cloud has no color information!")

    print("Loading RGB point cloud (for visualization)...")
    pcd_rgb = o3d.io.read_point_cloud(rgb_ply_path)
    if not pcd_rgb.has_colors():
        raise ValueError("RGB point cloud has no color information!")

    points_semantic = np.asarray(pcd_semantic.points)
    colors_semantic = np.asarray(pcd_semantic.colors)
    points_rgb = np.asarray(pcd_rgb.points)
    colors_rgb = np.asarray(pcd_rgb.colors)

    target_color = np.array(grass_color) / 255.0
    color_diff = np.linalg.norm(colors_semantic - target_color, axis=1)
    grass_mask = color_diff < color_tol
    grass_points = points_semantic[grass_mask]

    if len(grass_points) == 0:
        print("No grass points found with the specified color.")
        return

    print(f"Found {len(grass_points)} grass points")

    y_vals = grass_points[:, 1]
    y_ground = np.median(y_vals)
    ground_mask = np.abs(grass_points[:, 1] - y_ground) < y_tolerance
    ground_points = grass_points[ground_mask]

    if len(ground_points) < 10:
        print("Not enough ground points.")
        return

    print(f"Ground level (Y): {y_ground:.3f} m")

    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(ground_points)
    pcd_clean, ind = pcd_temp.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    ground_points_clean = ground_points[ind]

    proj_2d = ground_points_clean[:, [0, 2]]

    tree_rgb = o3d.geometry.KDTreeFlann(pcd_rgb)
    grass_colors_real = np.zeros((len(ground_points_clean), 3))
    for i, pt in enumerate(ground_points_clean):
        [k, idx, _] = tree_rgb.search_knn_vector_3d(pt, 1)
        if k > 0:
            grass_colors_real[i] = colors_rgb[idx[0]]
        else:
            grass_colors_real[i] = [0.5, 0.5, 0.5]

    proj_2d_ds = downsample_2d(proj_2d, factor=downsample_2d_factor)
    if len(proj_2d_ds) < 3:
        print("Not enough points after downsampling.")
        return

    try:
        alpha_shape = alphashape.alphashape(proj_2d_ds, alpha=alpha)
        if hasattr(alpha_shape, "exterior"):
            boundary_coords = np.array(alpha_shape.exterior.coords)
        else:
            boundary_coords = np.array(alpha_shape.coords)
        print(f"Alpha shape generated with {len(boundary_coords)} boundary points.")
    except Exception as e:
        print(f"Alpha shape failed: {e}, falling back to ConvexHull")
        try:
            hull = ConvexHull(proj_2d_ds)
            boundary_coords = proj_2d_ds[hull.vertices]
        except:
            print("ConvexHull also failed.")
            return

    boundary_2d_raw = smooth_boundary(boundary_coords, tolerance=simplify_tolerance)
    if len(boundary_2d_raw) < 3:
        print("Boundary too short after simplification.")
        return

    x, y = boundary_2d_raw[:, 0], boundary_2d_raw[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    print(f"Grass area (concave): {area:.2f} m²")

    if show_2d:
        plt.figure(figsize=(12, 10))
        all_xz = points_rgb[:, [0, 2]]
        plt.scatter(all_xz[:, 0], all_xz[:, 1],
                   c=colors_rgb, s=1, alpha=0.6, label='All Points (Real Color)')
        closed_boundary = np.vstack([boundary_2d_raw, boundary_2d_raw[0]])
        plt.plot(closed_boundary[:, 0], closed_boundary[:, 1],
                color=boundary_color, linewidth=3, label='Grass Boundary')
        plt.xlabel("X (m)")
        plt.ylabel("Z (m)")
        plt.title(f"Top-down View with Grass Boundary\nArea: {area:.2f} m²")
        plt.axis('equal')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    if show_3d:
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(points_rgb)
        pcd_vis.colors = o3d.utility.Vector3dVector(colors_rgb)

        boundary_3d = np.column_stack([
            boundary_2d_raw[:, 0],
            np.full(len(boundary_2d_raw), y_ground),
            boundary_2d_raw[:, 1]
        ])
        thick_boundary = create_thick_boundary_line(
            boundary_3d,
            radius=boundary_radius,
            color=boundary_color
        )

        o3d.visualization.draw_geometries(
            [pcd_vis, thick_boundary],
            window_name="Grass Area - 3D View (Real Color + Thick Boundary)",
            width=1200,
            height=800
        )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(os.path.join(save_dir, "grass_boundary_3d.txt"),
                   boundary_3d, fmt="%.6f", header="X Y Z", comments="")
        print(f"Saved 3D boundary to: {os.path.join(save_dir, 'grass_boundary_3d.txt')}")

        with open(os.path.join(save_dir, "grass_area.txt"), "w") as f:
            f.write(f"Grass Area (concave): {area:.2f} m²\n")
            f.write(f"Ground Y Level: {y_ground:.3f} m\n")
        print(f"Saved area info to: {os.path.join(save_dir, 'grass_area.txt')}")

        if show_2d:
            plt.savefig(os.path.join(save_dir, "grass_boundary_2d.png"), dpi=150, bbox_inches='tight')

    return {
        "area": area,
        "y_ground": y_ground,
        "boundary_2d": boundary_2d_raw,
        "boundary_3d": boundary_3d,
        "num_ground_points": len(ground_points_clean)
    }


if __name__ == "__main__":
    id = '44e7be4ce2'
    #id = 'lidar_data/0a15114e20'
    semantic_ply = f"semantic_results/fused_pointcloud_{id}.ply"
    rgb_ply = f"semantic_results/fused_pointcloud_{id}_rgb.ply"
    output_dir = f"rsemantic_results/grass_results_final_{id}"

    result = analyze_grass_area(
        semantic_ply_path=semantic_ply,
        rgb_ply_path=rgb_ply,
        grass_color=(4, 250, 7),
        color_tol=0.1,
        y_tolerance=0.3,
        alpha=1.5,
        downsample_2d_factor=0.02,
        simplify_tolerance=0.05,
        boundary_radius=0.05,
        boundary_color=[0, 1, 0],
        show_2d=True,
        show_3d=True,
        save_dir=output_dir
    )