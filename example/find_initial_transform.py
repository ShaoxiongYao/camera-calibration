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

source_pcd = o3d.io.read_point_cloud("data/test1001/pcd_camera2_000.ply")
target_pcd = o3d.io.read_point_cloud("data/test1001/pcd_robot_000.ply")

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
np.save('initial_transform2.npy', np.array(init_guess))
print("Initial guess: \n", init_guess)