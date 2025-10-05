import os
import sys
import glob
import numpy as np
import open3d as o3d
from argparse import ArgumentParser
from time import sleep

from franky import Affine, Robot

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

# initial guess, for debugging
robot2camera = np.array(
[[-0.9982791154718436, 0.056723603175950495, -0.014873674922674888, 0.2478944017758732],
 [0.0417451927617397, 0.5092786265950666, -0.8595886293257734, -0.10351870234525531],
 [-0.041184118750652234, -0.8587302850228856, -0.5107701703282619, 1.2855419353422566],
 [0.0, 0.0, 0.0, 1.0]]
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    args = parser.parse_args()

    data_dir = "data/test1001"

    # third-view camera
    camera1 = RealSenseCamera("247122073441", "D435i", {}) # right most camera
    camera2 = RealSenseCamera("242322073615", "D435i", {}) # middle camera

    # camera = KinectCamera()

    camera1.update()
    color1, depth1 = camera1.latest_rgbd_images()
    print(color1.shape, depth1.shape, depth1.max(), depth1.min())
    camera2.update()
    color2, depth2 = camera2.latest_rgbd_images()
    print(color2.shape, depth2.shape, depth2.max(), depth2.min())
    np.save('camera1_rgb.npy', color1)
    np.save('camera1_depth.npy', depth1)
    np.save('camera2_rgb.npy', color2)
    np.save('camera2_depth.npy', depth2)
    
    camera1.close()
    camera2.close()
    exit()

    robot_urdf = "/home/ydu/haowen/PAG/assets/franka_fr3_urdf/robot.urdf"
    world = WorldModel()
    world.loadElement(robot_urdf)  #load the robot model here
    robot = world.robot(0)

    index = glob.glob(f"{data_dir}/pcd_camera_*.ply")
    index = max([int(i.split('_')[-1].rsplit('.')[0]) for i in index])+1 if index else 0
    
    while input("Press 'enter' to capture a point cloud, 'q' to quit: ") != 'q':
        
        real_robot = Robot(args.host)
        state = real_robot.state
        robot.setConfig(robot.configFromDrivers(state.q))
        
        camera1.update()
        color1, depth1 = camera1.latest_rgbd_images()
        pc1 = camera1.latest_point_cloud()
        
        camera2.update()
        color2, depth2 = camera2.latest_rgbd_images()
        pc2 = camera2.latest_point_cloud()


        # state = real_robot.state
        # robot.setConfig(robot.configFromDrivers(state.q))

        # visualize pc, pc_color
        pc_klampt1 = PointCloud()
        pc_klampt1.setPoints(pc1[:, :3])
        pc_klampt1.addProperty('r',pc1[:, 3])
        pc_klampt1.addProperty('g',pc1[:, 4])
        pc_klampt1.addProperty('b',pc1[:, 5])

        pc_o3d1 = to_open3d(pc_klampt1)
        o3d.io.write_point_cloud(f"{data_dir}/pcd_camera1_{index:0>3}.ply", pc_o3d1)
        
        pc_klampt2 = PointCloud()
        pc_klampt2.setPoints(pc2[:, :3])
        pc_klampt2.addProperty('r',pc2[:, 3])
        pc_klampt2.addProperty('g',pc2[:, 4])
        pc_klampt2.addProperty('b',pc2[:, 5])   
        pc_o3d2 = to_open3d(pc_klampt2)
        o3d.io.write_point_cloud(f"{data_dir}/pcd_camera2_{index:0>3}.ply", pc_o3d2)

        robot_o3d = sample_robot_surface_points(robot_urdf, state.q, 10000)

        # robot_o3d.transform(robot2camera)
        o3d.visualization.draw_geometries([pc_o3d1])
        o3d.visualization.draw_geometries([pc_o3d2])
        o3d.visualization.draw_geometries([robot_o3d])

        # save pcd
        o3d.io.write_point_cloud(f"{data_dir}/pcd_robot_{index:0>3}.ply", robot_o3d)
        index += 1

    camera1.close()
    camera2.close()

