import os
import sys
import glob
import numpy as np
import open3d as o3d
from argparse import ArgumentParser
from time import sleep
from copy import deepcopy

from franky import Affine, Robot
from pathlib import Path

from realsense import RealSenseCamera

import klampt
from klampt import WorldModel,PointCloud
from klampt.math import vectorops,se3,so3
from klampt.model import ik, create
from klampt import vis
from klampt.io.open3d_convert import from_open3d, to_open3d
from klampt.io.numpy_convert import from_numpy, to_numpy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from calibrator.utils import sample_robot_surface_points

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    args = parser.parse_args()

    # data_dir = "data/test1025refine"
    # Path(data_dir).mkdir(parents=True, exist_ok=True)

    # third-view camera
    camera1 = RealSenseCamera("247122073441", "D435i", {}) # right most camera
    camera2 = RealSenseCamera("247122072484", "D435i", {}) # back camera

    robot_urdf = "/home/ydu/haowen/real2sim/assets/franka_fr3_urdf/robot.urdf"
    world = WorldModel()
    world.loadElement(robot_urdf)  #load the robot model here
    robot = world.robot(0)

    # cam1_robot2camera = np.linalg.inv(np.loadtxt('data/test1025/optimized_transform1_1021_1936.txt'))
    cam1_robot2camera = np.linalg.inv(np.loadtxt('/home/ydu/haowen/real2sim/cam_utils/optimized_transform1_0103_1751.txt'))
    cam1_camera2robot = np.linalg.inv(cam1_robot2camera)
    # cam2_robot2camera = np.linalg.inv(np.loadtxt('data/test1025/optimized_transform2_1026_1619.txt'))
    cam2_robot2camera = np.linalg.inv(np.loadtxt('/home/ydu/haowen/real2sim/cam_utils/optimized_transform2_0103_1824.txt'))
    cam2_camera2robot = np.linalg.inv(cam2_robot2camera)

    o3d_mesh = o3d.io.read_triangle_mesh('/home/ydu/haowen/real2sim/assets/franka_mujoco/meshes/finger_0.obj')
    o3d_mesh.compute_vertex_normals()

    vis.show()

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    
    center1 = np.array([0.0, -0.01773615, 0.07102775])
    center2 = np.array([0.0, 0.01773615, 0.07102775])

    initial_width = 0.055
    center1[1] -= initial_width / 2.0
    center2[1] += initial_width / 2.0

    box1 = create.box(0.06, 0.02, 0.1, center=center1.tolist())
    box1_geom = klampt.Geometry3D()
    box1_geom.set(box1)
    box2 = create.box(0.06, 0.02, 0.1, center=center2.tolist())
    box2_geom = klampt.Geometry3D()
    box2_geom.set(box2)

    # mpm_points = np.load('/home/ydu/haowen/real2sim/data/1104_sand_9/blue playdoh_mpm_points.npy')
    # mpm_pcd = from_numpy(mpm_points, 'PointCloud')
    # vis.add("mpm_pcd", mpm_pcd)

    # print('mpm points center:', np.mean(mpm_points, axis=0))
    # mpm_ball = create.sphere(0.02, center=np.mean(mpm_points, axis=0).tolist())
    # mpm_ball_geom = klampt.Geometry3D()
    # mpm_ball_geom.set(mpm_ball)
    # vis.add("mpm_ball", mpm_ball_geom)
    # vis.setColor("mpm_ball", 1.0, 0.0, 1.0, 0.5)

    # vis.debug("box1", box1, "box2", box2, from_open3d(coord_frame))

    try:
        while True:

            real_robot = Robot(args.host)
            state = real_robot.state
            robot.setConfig(robot.configFromDrivers(state.q))

            gripper_hmat = state.O_T_EE.matrix

            # print("Gripper pose: ", gripper_hmat)

            box1_geom.setCurrentTransform(*se3.from_ndarray(gripper_hmat))
            box2_geom.setCurrentTransform(*se3.from_ndarray(gripper_hmat))

            vis.add("left_finger", box1_geom)
            vis.setColor("left_finger", 0.0, 1.0, 0.0, 1.0)
            vis.add("right_finger", box2_geom)
            vis.setColor("right_finger", 0.0, 0.0, 1.0, 1.0)

            vis_coord_frame = deepcopy(coord_frame)
            vis_coord_frame.transform(gripper_hmat)
            vis.add("coord_frame", from_open3d(vis_coord_frame))
            vis.setColor("coord_frame", 1.0, 0.0, 0.0, 1.0)
            
            camera1.update()
            color1, depth1 = camera1.latest_rgbd_images()
            pc1 = camera1.latest_point_cloud()
            
            camera2.update()
            color2, depth2 = camera2.latest_rgbd_images()
            pc2 = camera2.latest_point_cloud()

            # visualize pc, pc_color
            pc_klampt1 = PointCloud()
            pc_klampt1.setPoints(pc1[:, :3])
            pc_klampt1.addProperty('r',pc1[:, 3])
            pc_klampt1.addProperty('g',pc1[:, 4])
            pc_klampt1.addProperty('b',pc1[:, 5])
            pc_klampt1.transform(*se3.from_ndarray(cam1_camera2robot))
            # pc_o3d1 = to_open3d(pc_klampt1)
            
            pc_klampt2 = PointCloud()
            pc_klampt2.setPoints(pc2[:, :3])
            pc_klampt2.addProperty('r',pc2[:, 3])
            pc_klampt2.addProperty('g',pc2[:, 4])
            pc_klampt2.addProperty('b',pc2[:, 5])   
            pc_klampt2.transform(*se3.from_ndarray(cam2_camera2robot))
            # pc_o3d2 = to_open3d(pc_klampt2)

            robot_o3d = sample_robot_surface_points(robot_urdf, state.q, 10000)
            robot_klp = from_open3d(robot_o3d)

            vis.add("pc1", pc_klampt1)
            vis.add("pc2", pc_klampt2)
            vis.add("robot", robot_klp)

            # save pcd
            # index1 += 1
            # index2 += 1

    except KeyboardInterrupt:
        camera1.close()
        camera2.close()

