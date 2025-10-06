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
from klampt.model.create import *
from klampt.model.calibrate import *
from klampt.math import vectorops,se3,so3
from klampt.model import ik
from klampt import vis
from klampt.io.open3d_convert import from_open3d, to_open3d

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from calibrator.utils import sample_robot_surface_points


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    parser.add_argument("--data_dir", type=str, default="data/test1005", help="data directory")
    args = parser.parse_args()

    data_dir = "data/test1005"
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    camera1_to_robot = np.loadtxt(f'{args.data_dir}/optimized_transform1_1005_1451.txt')
    camera2_to_robot = np.loadtxt(f'{args.data_dir}/optimized_transform2.txt')

    # third-view camera
    camera1 = RealSenseCamera("247122073441", "D435i", {}) # right most camera
    camera2 = RealSenseCamera("247122072484", "D435i", {}) # back camera

    robot_urdf = "/home/ydu/haowen/PAG/assets/franka_fr3_urdf/robot.urdf"
    world = WorldModel()
    world.loadElement(robot_urdf)  #load the robot model here
    robot = world.robot(0)
    
    # uncomment to save a frame of data
    # sleep(5) # wait for camera to warm up
    # camera1.update()
    # color1, depth1 = camera1.latest_rgbd_images()
    # print(color1.shape, depth1.shape, depth1.max(), depth1.min())
    # camera2.update()
    # color2, depth2 = camera2.latest_rgbd_images()
    # print(color2.shape, depth2.shape, depth2.max(), depth2.min())
    # np.save('../../real2sim/camera1_rgb.npy', color1)
    # np.save('../../real2sim/camera1_depth.npy', depth1)
    # np.save('../../real2sim/camera2_rgb.npy', color2)
    # np.save('../../real2sim/camera2_depth.npy', depth2)
    # exit()

    while input("Press 'enter' to view a point cloud, 'q' to quit: ") != 'q':

        real_robot = Robot(args.host)
        state = real_robot.state
        robot.setConfig(robot.configFromDrivers(state.q))
        
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
        pc_o3d1 = to_open3d(pc_klampt1)
        pc_o3d1.transform(camera1_to_robot)
        
        pc_klampt2 = PointCloud()
        pc_klampt2.setPoints(pc2[:, :3])
        pc_klampt2.addProperty('r',pc2[:, 3])
        pc_klampt2.addProperty('g',pc2[:, 4])
        pc_klampt2.addProperty('b',pc2[:, 5])   
        pc_o3d2 = to_open3d(pc_klampt2)
        pc_o3d2.transform(camera2_to_robot)

        robot_o3d = sample_robot_surface_points(robot_urdf, state.q, 10000)
        o3d.visualization.draw_geometries([pc_o3d1, pc_o3d2, robot_o3d])
        
        # save full pcd
        o3d.io.write_point_cloud(f"{data_dir}/pcd_transformed1.ply", pc_o3d1)
        o3d.io.write_point_cloud(f"{data_dir}/pcd_transformed2.ply", pc_o3d2)


    camera1.close()
    camera2.close()

