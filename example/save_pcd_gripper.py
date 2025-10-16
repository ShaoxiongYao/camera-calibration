import os
import sys
import glob
import numpy as np
import open3d as o3d
from argparse import ArgumentParser
import time

from franky import Affine, Robot
from pathlib import Path
from klampt.math import se3
from copy import deepcopy

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

import mujoco
import numpy as np

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    args = parser.parse_args()

    data_dir = "data/test_gripper"
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    robot_urdf = "/home/ydu/haowen/real2sim/assets/franka_fr3_urdf/robot.urdf"
    world = WorldModel()
    world.loadElement(robot_urdf)  #load the robot model here
    robot = world.robot(0)

    gripper_mesh = o3d.io.read_triangle_mesh("gripper_meshes.ply")

    # vis.add("world", world)
    # vis.show()

    real_robot = Robot(args.host)

    while True:
        state = real_robot.state
        # print("\nPose: ", real_robot.current_pose)
        # print("O_TT_E: ", state.O_T_EE)

        robot.setConfig(robot.configFromDrivers(state.q))

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
        coord_frame.transform(state.O_T_EE.matrix)
        # vis.add("coord_frame", from_open3d(coord_frame))

        gripper_link = robot.link('panda_hand')
        gripper_klampt_mesh = gripper_link.geometry()
        vis_gripper_klampt_mesh = to_open3d(gripper_klampt_mesh)
        o3d.visualization.draw_geometries([coord_frame, vis_gripper_klampt_mesh])

        vis_gripper_mesh = deepcopy(gripper_mesh)
        gripper_link = robot.link('panda_hand')
        gripper_hmat = se3.ndarray(gripper_link.getTransform())
        vis_gripper_mesh.transform(gripper_hmat)

        # vis.add("gripper_mesh", from_open3d(vis_gripper_mesh))

        time.sleep(1.0)
