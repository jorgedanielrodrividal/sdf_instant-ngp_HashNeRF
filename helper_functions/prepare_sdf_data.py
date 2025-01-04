import numpy as np
import open3d as o3d

def prepare_sdf_data(mesh_path, num_points=1000):
    """
    Prepares SDF data by computing points and their signed distances from the mesh.

    Parameters:
    mesh_path (str): Path to the mesh file.
    num_points (int): Number of random points to sample within the mesh's bounding box.

    Returns:
    dict: A configuration dictionary with points, sdf_values, and other parameters.
    """
    # Load mesh and compute bounding box
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    bbox = mesh.get_axis_aligned_bounding_box()

    # Sample random points within the bounding box
    points_random = np.random.uniform(bbox.min_bound, bbox.max_bound, size=(num_points, 3))

    # Compute SDF values using raycasting
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    points_tensor = o3d.core.Tensor(points_random, dtype=o3d.core.Dtype.Float32)

    distances = scene.compute_distance(points_tensor)
    inside = distances.numpy() < 1e-3  # Tolerance for "inside"

    #sdf_values = distances.copy()
    #sdf_values[distances < 1e-3] *= -1  # Negative for points inside the mesh

    # Count the number of points inside and outside
    num_inside = np.sum(inside)  # Number of points inside the mesh
    num_outside = len(inside) - num_inside  # Number of points outside the mesh

    # Calculate SDF values
    sdf_values = distances.numpy()  # Distances from points to the mesh surface
    sdf_values[inside] *= -1  # Negative SDF for inside points

    # Print results
    #print(f"Number of points inside the mesh: {num_inside}")
    #print(f"Number of points outside the mesh: {num_outside}")
    #print(f"SDF Value Range: Min {sdf_values.min()}, Max {sdf_values.max()}")

    # Return configuration dictionary
    return {
        "points_random": points_random,
        "sdf_values": sdf_values,
           }

result = prepare_sdf_data(mesh_path)
points_random = result["points_random"]
sdf_values = result["sdf_values"]
