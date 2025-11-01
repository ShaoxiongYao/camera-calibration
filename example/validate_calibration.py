import os
import sys
import glob
import numpy as np
import open3d as o3d
from argparse import ArgumentParser
from time import sleep

from franky import Affine, Robot
from pathlib import Path

from realsense import RealSenseCamera

import klampt
from klampt import WorldModel,PointCloud
from klampt.math import vectorops,se3,so3
from klampt.model import ik
from klampt import vis
from klampt.io.open3d_convert import from_open3d, to_open3d

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from calibrator.utils import sample_robot_surface_points

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    args = parser.parse_args()

    data_dir = "data/test1025refine"
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # third-view camera
    camera1 = RealSenseCamera("247122073441", "D435i", {}) # right most camera
    camera2 = RealSenseCamera("247122072484", "D435i", {}) # back camera

    robot_urdf = "/home/ydu/haowen/real2sim/assets/franka_fr3_urdf/robot.urdf"
    world = WorldModel()
    world.loadElement(robot_urdf)  #load the robot model here
    robot = world.robot(0)

    # cam1_robot2camera = np.linalg.inv(np.loadtxt('data/test1025/optimized_transform1_1021_1936.txt'))
    cam1_robot2camera = np.linalg.inv(np.loadtxt('/home/ydu/haowen/real2sim/cam_utils/optimized_transform0_1021.txt'))
    cam1_camera2robot = np.linalg.inv(cam1_robot2camera)
    # cam2_robot2camera = np.linalg.inv(np.loadtxt('data/test1025/optimized_transform2_1026_1619.txt'))
    cam2_robot2camera = np.linalg.inv(np.loadtxt('/home/ydu/haowen/real2sim/cam_utils/optimized_transform1_1026.txt'))
    cam2_camera2robot = np.linalg.inv(cam2_robot2camera)

    tmesh2 = o3d.io.read_triangle_mesh('/home/ydu/haowen/real2sim/data/1030_push_3/pringles_scaled.obj')
    trans_mat2 = np.loadtxt('/home/ydu/haowen/real2sim/data/1030_push_3/pringles_mujoco_cam1.txt')
    tmesh2.transform(trans_mat2)
    vis.add("target_obj2", from_open3d(tmesh2))

    tmesh3 = o3d.io.read_triangle_mesh('/home/ydu/haowen/real2sim/data/1030_push_3/pringles_scaled.obj')
    int_mat2 = np.loadtxt('/home/ydu/haowen/real2sim/data/1030_push_3/pringles_6d_cam1.txt')
    tmesh3.transform(int_mat2)
    tmesh3.transform(cam2_camera2robot)
    tmesh3.paint_uniform_color([1,0,0])
    vis.add("target_obj3", from_open3d(tmesh3))


    tmesh = o3d.io.read_triangle_mesh('/home/ydu/haowen/real2sim/data/1030_push_3/white coconut milk carton_scaled.obj')
    trans_mat = np.loadtxt('/home/ydu/haowen/real2sim/data/1030_push_3/white coconut milk carton_mujoco_cam1.txt')
    tmesh.transform(trans_mat)
    vis.add("target_obj", from_open3d(tmesh))

    vis.show()

    index1 = glob.glob(f"{data_dir}/pcd_camera1_*.ply")
    index1 = max([int(i.split('_')[-1].rsplit('.')[0]) for i in index1])+1 if index1 else 0
    index2 = glob.glob(f"{data_dir}/pcd_camera2_*.ply")
    index2 = max([int(i.split('_')[-1].rsplit('.')[0]) for i in index2])+1 if index2 else 0
    assert index1 == index2, "camera1 and camera2 index should be the same"
    
    try:
        while True:

            real_robot = Robot(args.host)
            state = real_robot.state
            robot.setConfig(robot.configFromDrivers(state.q))
            
            camera1.update()
            color1, depth1 = camera1.latest_rgbd_images()
            pc1 = camera1.latest_point_cloud()
            
            camera2.update()
            color2, depth2 = camera2.latest_rgbd_images()
            pc2 = camera2.latest_point_cloud()

            np.save('tmp_pc2.npy', pc2)
            input()

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

