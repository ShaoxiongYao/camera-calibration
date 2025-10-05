import numpy as np
import time
import warnings
from camera_apis import RGBDCamera,CameraSettings,RGBDCameraSettings
from dataclasses import dataclass, fields
import dacite
from typing import Optional
try:
    import pyrealsense2 as rs
    HAS_RS2 = True
except ImportError:
    HAS_RS2 = False

@dataclass
class LaserSensorSettings:
    laser_power : Optional[float] = None
    min_distance : Optional[float] = None
    receiver_gain : Optional[float] = None
    noise_filtering : Optional[float] = None
    pre_processing_sharpening : Optional[float] = None
    post_processing_sharpening : Optional[float] = None
    confidence_threshold : Optional[float] = None

@dataclass
class RealSenseCameraSettings(RGBDCameraSettings):
    serial_num: str = ''
    camera_type: str = ''
    laser_settings : Optional[LaserSensorSettings] = None

    def __init__(self, serial_num : str, camera_type : str, extra_settings = None):
        self.serial_num = serial_num
        self.camera_type = camera_type

        if camera_type in ['SR','D405','D435i']:
            rgb = CameraSettings(width=640, height=480, rate=30, color=True, color_format='rgb8')
            depth = CameraSettings(width=640, height=480, rate=30, color=False)
            RGBDCameraSettings.__init__(self, rgb=rgb, depth=depth, depth_aligned=True, depth_scale=1.0, depth_min=0.5, depth_max=5.0)
        elif camera_type == 'L515':
            rgb = CameraSettings(width=1280, height=720, rate=30, color=True, color_format='rgb8')
            depth = CameraSettings(width=640, height=480, rate=30, color=False)
            RGBDCameraSettings.__init__(self, rgb=rgb, depth=depth, depth_aligned=True, depth_scale=1.0, depth_min=0.5, depth_max=8.0)
        else:
            print('\n SensorModule: Failed to initialize camera with serial number {} due to unsuported camera type {}. Please review the camera_settings_file and try again.'.format(serial_num,camera_type))
            raise TypeError('Non-supported camera type selected found!')
        
        if extra_settings is not None:
            self.laser_settings = LaserSensorSettings(**extra_settings)
            

class RealSenseCamera(RGBDCamera):
    """Driver for a realsense camera. Handles L515s and SRs.

    REMARK: `update()` can only be called in the same process in
        which this driver was instantiated, otherwise, it will
        runtime error saying frames did not arrive for 5000 ms.
    """
    def __init__(self, serial_num : str, camera_type : str = '', extra_settings=None):
        if isinstance(serial_num, RealSenseCameraSettings):
            self._settings = serial_num
        elif isinstance(serial_num, dict):
            self._settings = RealSenseCameraSettings(serial_num['serial_num'],serial_num['camera_type'])
        else:
            self._settings = RealSenseCameraSettings(serial_num,camera_type,**extra_settings)
        assert self._settings.rgb.color_format == 'rgb8'

        if not HAS_RS2:
            warnings.warn("Realsense camera data is not available because the pyrealsense2 module is not installed")
            self.pipeline = None
            return

        EXTRA_SETTING_TO_RS_ID = {
            'laser_power':rs.option.laser_power,
            'min_distance':rs.option.min_distance,
            'receiver_gain':rs.option.receiver_gain,
            'noise_filtering':rs.option.noise_filtering,
            'pre_processing_sharpening':rs.option.pre_processing_sharpening,
            'post_processing_sharpening':rs.option.post_processing_sharpening,
            'confidence_threshold':rs.option.confidence_threshold
        }
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.latest_aligned_frame = None
        self.frame_time = None
        try:
            self.config.enable_device(self._settings.serial_num.encode('utf-8'))
            self.config.enable_stream(
                rs.stream.depth, self._settings.depth.width, self._settings.depth.height, rs.format.z16, self._settings.depth.rate)
            self.config.enable_stream(
                rs.stream.color, self._settings.rgb.width, self._settings.rgb.height, rs.format.rgb8, self._settings.rgb.rate)
            
            self.numpix = self._settings.rgb.width*self._settings.rgb.height
            self.align_to = rs.stream.color
            self.align = rs.align(self.align_to)
            # Start streaming
            self.profile = self.pipeline.start(self.config)
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()
            if self._settings.laser_settings is not None:
                print("Starting realsense L515 with extra settings")
                for f in fields(self._settings.laser_settings):
                    value = getattr(self._settings.laser_settings,f.name)
                    if value is not None:
                        self.depth_sensor.set_option(EXTRA_SETTING_TO_RS_ID[f.name],value)
                        print(f.name,"set to",self.depth_sensor.get_option(f.name))
            if self._settings.rgb.intrinsics is None:
                print("Updating Realsense intrinsics from factory settings")
                self.update_intrinsics_from_profile()
            # we sleep for 3 seconds to stabilize the color image - no idea why, but if we query it soon after starting, color image is distorted.
            self.pc = rs.pointcloud()
            
            # print the intrinsics
            print("Realsense rgb intrinsics:")
            print(self._settings.rgb.intrinsics)
            print("Realsense depth intrinsics:")
            print(self._settings.depth.intrinsics)
        except Exception as e:
            print(e, 'Invalid camera serial number?')
            self.pipeline.stop()
            self.pipeline = None

    def connected(self):
        return self.pipeline is not None

    def update(self):
        if self.pipeline is None:
            return False
        frames = self.pipeline.wait_for_frames()

        self.frame_time = time.time()
        self.pc_time = self.frame_time
        if not frames.get_depth_frame() or not frames.get_color_frame():
            return False
        # Fetch color and depth frames and align them
        aligned_frames = self.align.process(frames)
        #depth_frame = np.asarray(aligned_frames.get_depth_frame())
        #color_frame = aligned_frames.get_color_frame()
        self.latest_capture_time = time.time()
        self.latest_aligned_frame = aligned_frames
        return True

    def last_update(self):
        return self.frame_time

    def channels(self):
        return ['rgb', 'depth', 'point_cloud']
    
    def value(self, id):
        if self.pipeline is None:
            return None
        if self.latest_aligned_frame is None:
            return None
        if id is None:
            res = {}
            for ch in self.channels():
                res[ch] = self.value(ch)
            return res
        if id == 'point_cloud':
            return self.latest_point_cloud()
        elif id == 'rgb':
            color_frame = self.latest_aligned_frame.get_color_frame()
            return np.array(color_frame.get_data())
        elif id == 'depth':
            depth_frame = self.latest_aligned_frame.get_depth_frame()

            tmp_depth = np.array(depth_frame.get_data()).astype(float)
            return tmp_depth*self.depth_scale
            #depth_mm = (tmp_depth*1000).astype(np.uint16)

    def latest_rgbd_images(self):
        """
        Returns the color and depth images from the last frame.

        Args:
        Returns:
            color : returns the color image in the camera's local frame
            as a numpy array HxWx3 RGB image, or None if the data is not available
            depth : returns the depth image in the camera's local frame
            as a numpy array HxW depth image in meters, or None if the data is not available
        """
        if self.pipeline is None:
            return None, None
        if self.latest_aligned_frame is None:
            return None, None
        # Fetch color and depth frames and align them
        depth_frame = self.latest_aligned_frame.get_depth_frame()
        color_frame = self.latest_aligned_frame.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        color = np.array(color_frame.get_data())
        tmp_depth = np.array(depth_frame.get_data()).astype(float)
        depth = tmp_depth*self.depth_scale
        return color, depth

    def latest_point_cloud(self):
        """
        Returns the point cloud from the last frame.

        Args:
        Returns:
            pc : returns the point cloud in the camera's local frame
            as a numpy array XYZRGB PointCloud, or None if the data is not available
        """
        if self.pipeline is None:
            return None
        if self.latest_aligned_frame is None:
            return None
        # Fetch color and depth frames and align them
        depth_frame = self.latest_aligned_frame.get_depth_frame()
        color_frame = self.latest_aligned_frame.get_color_frame()

        # Tell pointcloud object to map to this color frame
        self.pc.map_to(color_frame)
        # Generate the pointcloud and texture mappings
        points = self.pc.calculate(depth_frame)
        vtx = np.asarray(points.get_vertices())
        pure_point_cloud = np.zeros((self.numpix, 3))
        pure_point_cloud[:, 0] = vtx['f0']
        pure_point_cloud[:, 1] = vtx['f1']
        pure_point_cloud[:, 2] = vtx['f2']
        color_t = np.asarray(color_frame.get_data()).reshape(self.numpix, 3)/255
        return np.hstack((pure_point_cloud,color_t))
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(pure_point_cloud)
        # point_cloud.colors = o3d.utility.Vector3dVector(color_t)
        # return point_cloud

    def close(self):
        if self.pipeline is None:
            return
        print('Safely closing Realsense camera', self._settings.serial_num)
        self.pipeline.stop()
        self.pipeline = None

    def update_intrinsics_from_profile(self):
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        if color_intrinsics.width != self._settings.rgb.width or color_intrinsics.height != self._settings.rgb.height:
            print("Warning, color intrinsics do not match settings")
        if color_intrinsics.fx == 0:
            print("Camera",self._settings.camera_type,"does not provide color intrinsics")
            return
        self._settings.rgb.intrinsics = {'fx': color_intrinsics.fx, 'fy': color_intrinsics.fy, 'cx': color_intrinsics.ppx, 'cy': color_intrinsics.ppy}
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        if depth_intrinsics.width != self._settings.depth.width or depth_intrinsics.height != self._settings.depth.height:
            print("Warning, color intrinsics do not match settings")
        if depth_intrinsics.fx == 0:
            print("Camera",self._settings.camera_type,"does not provide depth intrinsics")
            return
        self._settings.depth.intrinsics = {'fx': depth_intrinsics.fx, 'fy': depth_intrinsics.fy, 'cx': depth_intrinsics.ppx, 'cy': depth_intrinsics.ppy}
        return color_intrinsics,depth_intrinsics


def test_plot(cam : RealSenseCamera, count = 1):
    for i in range(count):
        cam.update()
        a = cam.value('rgb')
        b = cam.value('depth')
        # b = cam.latest_point_cloud()
        from matplotlib import pyplot as plt
        fig,axs = plt.subplot(1,2,figsize=(10,5))
        axs[0].imshow(a)
        axs[1].imshow(b)
        plt.show()
        # time.sleep(0.1)

def simple_test():
    cam = RealSenseCamera("f0190400","L515",{
            "laser_power":0,
            "min_distance":0.0,
            "receiver_gain":18.0,
            "noise_filtering":2.0,
            "pre_processing_sharpening":2.0
            })
    test_plot(cam)

def multiprocess_test():
    import multiprocessing
    p = multiprocessing.Process(target=simple_test)
    p.start()
    p.join()

if __name__ == '__main__':
    print(help(rs.option))
    simple_test()
    