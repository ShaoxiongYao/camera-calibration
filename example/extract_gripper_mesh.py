import mujoco
import numpy as np
import argparse
import open3d as o3d


def extract_triangle_mesh(model, geom_id):
    """Extract triangle mesh data from a MuJoCo geometry."""
    geom_type = model.geom_type[geom_id]
    geom_dataid = model.geom_dataid[geom_id]
    
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = geom_dataid
        vert_start = model.mesh_vertadr[mesh_id]
        vert_num = model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[vert_start:vert_start + vert_num].copy()
        
        face_start = model.mesh_faceadr[mesh_id]
        face_num = model.mesh_facenum[mesh_id]
        faces = model.mesh_face[face_start:face_start + face_num].copy()
        
        return vertices, faces
    return None, None


def get_geom_transform(model, data, geom_id):
    """Get the rigid body transformation of a geometry."""
    position = data.geom_xpos[geom_id].copy()
    rotation_matrix = data.geom_xmat[geom_id].reshape(3, 3).copy()
    return position, rotation_matrix


def get_body_transform(model, data, body_id):
    """Get the rigid body transformation of a body."""
    position = data.xpos[body_id].copy()
    rotation_matrix = data.xmat[body_id].reshape(3, 3).copy()
    quaternion = data.xquat[body_id].copy()
    return position, rotation_matrix, quaternion


def apply_transform(mesh, position, rotation_matrix):
    """Apply rigid body transformation to mesh."""
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position
    mesh.transform(transform)
    return mesh


def visualize_gripper(xml_path, gripper_body_name='hand'):
    """Extract and visualize gripper meshes with Open3D."""
    
    # Load model
    print(f"Loading model from {xml_path}...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # Get gripper body
    try:
        gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gripper_body_name)
    except:
        print(f"Could not find body '{gripper_body_name}'")
        return
    
    # Get gripper base transform
    base_pos, base_rot, base_quat = get_body_transform(model, data, gripper_body_id)
    print(f"\nGripper Base Transform:")
    print(f"  Position: {base_pos}")
    print(f"  Quaternion [w,x,y,z]: {base_quat}")
    
    # Find all child bodies
    def get_child_bodies(body_id):
        children = [body_id]
        for i in range(model.nbody):
            if model.body_parentid[i] == body_id:
                children.extend(get_child_bodies(i))
        return children
    
    gripper_bodies = get_child_bodies(gripper_body_id)
    print(f"\nFound {len(gripper_bodies)} bodies in gripper tree")
    
    # Collect geometries
    geometries = []
    
    # Add coordinate frame at world origin
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    
    # Add gripper base frame
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    apply_transform(base_frame, base_pos, base_rot)
    geometries.append(base_frame)
    
    # Colors for different meshes
    colors = [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8], 
              [0.8, 0.8, 0.2], [0.8, 0.2, 0.8], [0.2, 0.8, 0.8]]
    color_idx = 0
    
    # Extract meshes
    for geom_id in range(model.ngeom):
        geom_body_id = model.geom_bodyid[geom_id]
        
        if geom_body_id in gripper_bodies:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            vertices, faces = extract_triangle_mesh(model, geom_id)
            
            if vertices is not None:
                print(f"  {geom_name}: {len(vertices)} vertices, {len(faces)} faces")
                
                # Create Open3D mesh
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color(colors[color_idx % len(colors)])
                
                # Apply transform
                geom_pos, geom_rot = get_geom_transform(model, data, geom_id)
                apply_transform(mesh, geom_pos, geom_rot)
                
                geometries.append(mesh)
                color_idx += 1
    
    print(f"\nVisualizing {color_idx} meshes...")

    sum_mesh = sum(geometries[2:], o3d.geometry.TriangleMesh())  # Skip coordinate frames
    o3d.io.write_triangle_mesh("gripper_meshes.ply", sum_mesh)  # Save the first mesh as an example
    
    # Visualize
    o3d.visualization.draw_geometries(geometries, 
                                       window_name="Gripper Mesh Visualization",
                                       width=1024, height=768)


def main():
    parser = argparse.ArgumentParser(description='Visualize gripper mesh with Open3D')
    parser.add_argument('--xml_path', type=str, 
                        default='../../real2sim/assets/franka_mujoco/franka_gripper.xml',
                        help='Path to MuJoCo XML file')
    parser.add_argument('--gripper_body', type=str, default='hand',
                        help='Name of the gripper base body')
    args = parser.parse_args()
    
    try:
        visualize_gripper(args.xml_path, args.gripper_body)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()