import os
import sys
import glob
import numpy as np
import open3d as o3d

from iml_internal.hardware.cameras.real_sense import RealSenseCamera
from iml_internal.hardware.cameras.azure import KinectCamera

import klampt
from klampt.model.create import *
from klampt.model.calibrate import *
from klampt.math import vectorops,se3,so3
from klampt.model import ik
from klampt import vis
from klampt.io.open3d_convert import from_open3d, to_open3d

import kortex_utils
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

    data_dir = "data"

    # third-view camera
    camera = RealSenseCamera("f0271386", "L515")
    # camera = KinectCamera()

    robot_urdf = "/home/motion/plant-model/robot_model/kinova_gen3_repaired.urdf"
    world = WorldModel()
    world.loadElement(robot_urdf)  #load the robot model here
    robot = world.robot(0)

    index = glob.glob(f"{data_dir}/pcd_camera_*.ply")
    index = max([int(i.split('_')[-1].split('.')[0]) for i in index])+1 if index else 0
    args = kortex_utils.parseConnectionArguments()
    with kortex_utils.KortexAsyncInterface(args) as interface:

        while input("Press 'enter' to capture a point cloud, 'q' to quit: ") != 'q':
            camera.update()
            color, depth = camera.latest_rgbd_images()
            pc = camera.latest_point_cloud()

            robot.setConfig([0]+list(interface.angles)+[0])

            # visualize pc, pc_color
            pc_klampt = PointCloud()
            pc_klampt.setPoints(pc[:, :3])
            pc_klampt.addProperty('r',pc[:, 3])
            pc_klampt.addProperty('g',pc[:, 4])
            pc_klampt.addProperty('b',pc[:, 5])

            pc_o3d = to_open3d(pc_klampt)
            robot_o3d = sample_robot_surface_points(robot_urdf, interface.angles, 10000)

            # robot_o3d.transform(robot2camera)
            o3d.visualization.draw_geometries([pc_o3d])

            # save pcd
            o3d.io.write_point_cloud(f"{data_dir}/pcd_camera_{index:0>3}.ply", pc_o3d)
            o3d.io.write_point_cloud(f"{data_dir}/pcd_robot_{index:0>3}.ply", robot_o3d)
            index += 1

    camera.safely_close()

