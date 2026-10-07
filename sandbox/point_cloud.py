import open3d as o3d
import numpy as np

# 1. Read the PLY point cloud
pcd = o3d.io.read_point_cloud("/Users/ali/Workspace/PhD/Cable/cable_simulation/results/Ethernet_tube_fit/centerline.ply")

# 2. Convert points to a standard NumPy array (if you need to manipulate raw data)
points = np.asarray(pcd.points)
print(f"Loaded {len(points)} points.")

# 3. Quick interactive 3D visualization 
o3d.visualization.draw_geometries([pcd])
