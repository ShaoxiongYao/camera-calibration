import os
import sys
import glob
import numpy as np
from copy import deepcopy
import open3d as o3d

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from calibrator.registration import register
from calibrator.refinement import optimize_registration

# initial guess
# L515
robot2camera = np.array(
[[-0.99828379, 0.05547536, -0.01876052, 0.2427629],
 [0.04394943, 0.49798846, -0.86606925, -0.09438848],
 [-0.03870298, -0.86540741, -0.49957192, 1.28361232],
 [0.0, 0.0, 0.0, 1.0]]
)

robot2camera = np.linalg.inv(np.load('initial_transform2.npy'))

# Kinect
# robot2camera = np.array(
# [[-9.99973584e-01, -7.25495077e-03, -4.44568382e-04, -5.65225358e-01],
#  [-2.33064713e-04, 9.31356531e-02, -9.95653401e-01, 4.17600780e-01],
#  [7.26482158e-03, -9.95626996e-01, -9.31348836e-02, 1.09268988e+00],
#  [0.0, 0.0, 0.0, 1.0]]
# )


camera2robot = np.linalg.inv(robot2camera)
voxel_size = 0.003

if __name__ == "__main__":
    data_dir = "data/test1001"

    # third-view camera
    index = glob.glob(f"{data_dir}/pcd_camera2_*.ply")
    index = max([int(i.split('_')[-1].split('.')[0]) for i in index])+1 if index else 0

    pc_camera = []
    pc_robot = []
    for i in range(index):
        cam = o3d.io.read_point_cloud(f"{data_dir}/pcd_camera2_{i:0>3}.ply")
        robot = o3d.io.read_point_cloud(f"{data_dir}/pcd_robot_{i:0>3}.ply")

        robot_in_cam = deepcopy(robot).transform(robot2camera)
        # bbox = robot_in_cam.get_axis_aligned_bounding_box()
        # padding = 0.05
        # bbox = o3d.geometry.AxisAlignedBoundingBox(
        #     min_bound=[bbox.min_bound[0]-padding, bbox.min_bound[1]-padding, bbox.min_bound[2]-padding],
        #     max_bound=[bbox.max_bound[0]+padding, bbox.max_bound[1]+padding, bbox.max_bound[2]+padding]
        # )
        
        o3d.visualization.draw_geometries([cam, robot_in_cam])

        # # use bbox to crop the point cloud
        # cam = cam.crop(bbox)

        # cam = cam.voxel_down_sample(voxel_size=voxel_size)
        # robot = robot.voxel_down_sample(voxel_size=voxel_size)

        # # select points close enough to the robot
        # import sklearn.neighbors
        # knn = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto')
        # knn.fit(np.asarray(robot_in_cam.points))
        # dist, idx = knn.kneighbors(np.asarray(cam.points))
        # idx = idx.reshape(-1)
        # to_keep = np.where(dist < 0.05)[0]
        # cam = cam.select_by_index(to_keep)

        pc_camera.append(cam)
        pc_robot.append(robot)

    X_robot_camera, error = optimize_registration(
        source_pcds=pc_camera,
        target_pcds=pc_robot,
        initial_transform=camera2robot,
        distance_threshold=0.10,
        max_iter=100,
        stop_threshold=0.001,
    )

    print("camera2robot: \n", X_robot_camera)
    print("robot2camera: \n", np.linalg.inv(X_robot_camera))

    print(error)

    # check the result
    for i in range(index):
        pc_camera[i].transform(X_robot_camera)
        o3d.visualization.draw_geometries([pc_camera[i], pc_robot[i]])
