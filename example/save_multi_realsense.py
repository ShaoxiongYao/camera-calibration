"""List all connected RealSense cameras and dump RGB, depth, and point cloud to disk.

Output layout:
    <output_dir>/<YYYYmmdd_HHMMSS>/<serial>/
        rgb.png
        depth.png     (uint16, millimeters)
        depth.npy     (float32, meters)
        pointcloud.ply
        info.txt
"""
import argparse
import os
import time
from datetime import datetime

import numpy as np
import pyrealsense2 as rs
from PIL import Image

from realsense import RealSenseCamera


SUPPORTED_TYPES = ('L515', 'D405', 'D435i', 'SR')


def infer_camera_type(device_name: str) -> str:
    name = device_name.upper()
    if 'L515' in name:
        return 'L515'
    if 'D405' in name:
        return 'D405'
    if 'D435' in name:
        return 'D435i'
    if 'SR' in name:
        return 'SR'
    raise ValueError(f"Cannot infer supported camera type from name '{device_name}'. "
                     f"Supported types: {SUPPORTED_TYPES}")


def enumerate_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    infos = []
    for dev in devices:
        infos.append({
            'name': dev.get_info(rs.camera_info.name),
            'serial': dev.get_info(rs.camera_info.serial_number),
            'firmware': dev.get_info(rs.camera_info.firmware_version)
                if dev.supports(rs.camera_info.firmware_version) else 'unknown',
        })
    return infos


def save_point_cloud_ply(path: str, pc: np.ndarray):
    """Write an XYZRGB point cloud (Nx6, colors in [0,1]) as ASCII PLY."""
    valid = np.isfinite(pc[:, :3]).all(axis=1) & (np.linalg.norm(pc[:, :3], axis=1) > 0)
    pc = pc[valid]
    xyz = pc[:, :3]
    rgb = np.clip(pc[:, 3:6] * 255.0, 0, 255).astype(np.uint8)

    with open(path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(xyz)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f'{x} {y} {z} {r} {g} {b}\n')


def capture_camera(serial: str, camera_type: str, out_dir: str, warmup_frames: int = 100):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Opening {camera_type} {serial}...")
    cam = RealSenseCamera(serial, camera_type, extra_settings={})
    if not cam.connected():
        print(f"  Failed to connect to {serial}")
        return False

    try:
        for _ in range(warmup_frames):
            cam.update()
            time.sleep(0.03)

        cam.update()
        color, depth = cam.latest_rgbd_images()
        pc = cam.latest_point_cloud()

        if color is None or depth is None:
            print(f"  No frames received from {serial}")
            return False

        Image.fromarray(color).save(os.path.join(out_dir, 'rgb.png'))
        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_mm).save(os.path.join(out_dir, 'depth.png'))
        np.save(os.path.join(out_dir, 'depth.npy'), depth.astype(np.float32))

        if pc is not None:
            save_point_cloud_ply(os.path.join(out_dir, 'pointcloud.ply'), pc)

        rgb_intr = cam._settings.rgb.intrinsics
        depth_intr = cam._settings.depth.intrinsics
        with open(os.path.join(out_dir, 'info.txt'), 'w') as f:
            f.write(f'serial: {serial}\n')
            f.write(f'camera_type: {camera_type}\n')
            f.write(f'rgb_shape: {color.shape}\n')
            f.write(f'depth_shape: {depth.shape}\n')
            f.write(f'rgb_intrinsics: {rgb_intr}\n')
            f.write(f'depth_intrinsics: {depth_intr}\n')

        print(f"  Saved to {out_dir}")
        return True
    finally:
        cam.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='data',
                        help='Root output directory (default: data)')
    parser.add_argument('--warmup-frames', type=int, default=15,
                        help='Number of frames to drop before saving (default: 15)')
    args = parser.parse_args()

    devices = enumerate_devices()
    if not devices:
        print("No RealSense cameras detected.")
        return

    print(f"Found {len(devices)} RealSense camera(s):")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d['name']}  serial={d['serial']}  fw={d['firmware']}")
    print()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    print(f"Writing session data to {session_dir}\n")

    for d in devices:
        try:
            camera_type = infer_camera_type(d['name'])
        except ValueError as e:
            print(f"Skipping {d['serial']}: {e}")
            continue
        capture_camera(d['serial'], camera_type,
                       os.path.join(session_dir, d['serial']),
                       warmup_frames=args.warmup_frames)


if __name__ == '__main__':
    main()
