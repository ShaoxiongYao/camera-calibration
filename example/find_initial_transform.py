import numpy as np
import open3d as o3d
from typing import List, Optional


def pick_correspondences(
    pcd: o3d.geometry.PointCloud,
) -> List[int]:
    """
    Pick correspondences between two point clouds.

    :param pcd: Point cloud to pick correspondences from.
    :return: List of indices of correspondences.
    """
    print(
        "\n1) Please pick at least three correspondences using [shift + left click]"
        "\n   Press [shift + right click] to undo point picking"
        "\n2) After picking points, press 'Q' to close the window\n"
    )
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()

data_dir = "data/test0102_2"
calib_frame_id = 5

for camera_index in [1, 2]:

    source_pcd = o3d.io.read_point_cloud(f"{data_dir}/pcd_camera{camera_index}_{calib_frame_id:03d}.ply")
    target_pcd = o3d.io.read_point_cloud(f"{data_dir}/pcd_robot_{calib_frame_id:03d}.ply")

    # manually select correspondences
    while True:
        source_points_indices = pick_correspondences(source_pcd)
        target_points_indices = pick_correspondences(target_pcd)
        if (
            (not len(source_points_indices) >= 3) or
            (len(source_points_indices) != len(target_points_indices))
        ): 
            print(
                "Correspondences not selected correctly. "
                "Please select at least three correspondences. \n"
                f"{len(source_points_indices)} source points selected, "
                f"{len(target_points_indices)} target points selected. \n"
                "Please try again. \n"
            )
        else:
            break

    # construct initial guess
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    init_guess = p2p.compute_transformation(
        source_pcd,
        target_pcd,
        o3d.utility.Vector2iVector(
            np.stack([
                source_points_indices,
                target_points_indices
            ], axis=1)
        )
    )
    np.save(f'{data_dir}/initial_transform{camera_index}.npy', np.array(init_guess))
    print("Initial guess: \n", init_guess)