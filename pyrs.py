import os
import json

import cv2
import pyrealsense2 as rs
import numpy as np


class PyRS:

    def __init__(self, w=640, h=480, depths=True, frame_rate=30):
        '''
        Initializing the Python RealSense Control Flow:
        w: Int (default = 640)
        h: Int (default = 480)
        depth: Bool (default = True)
        frame_rate: Int (default = 30)

        RGB and Depths formats are: bgr8, z16

        Note: In this class, variables should not be directly changed.
        '''
        self.depths_on = depths
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, frame_rate)
        self.intrinsic = None

        if depths:
            self.align = rs.align(rs.stream.color)
            self._preset = 1
            # Presets:
            # 0: Custom
            # 1: Default
            # 2: Hand 
            # 3: High Accuracy
            # 4: High Density
            # 5: Medium Density

            # depths interpolation
            self.interpolation = cv2.INTER_NEAREST  # use nearest neighbor
            # self.interpolation = cv2.INTER_LINEAR  # linear
            # self.interpolation = cv2.INTER_CUBIC  # cubic

            self._config.enable_stream(rs.stream.depth, w, h, rs.format.z16, frame_rate)
            self.colorizer = rs.colorizer()

            # initialize filters
            self.decimation = rs.decimation_filter()
            self.decimation.set_option(rs.option.filter_magnitude, 4)

            self.depths_to_disparity = rs.disparity_transform(True)

            self.spatial = rs.spatial_filter()
            self.spatial.set_option(rs.option.filter_magnitude, 5)
            self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial.set_option(rs.option.filter_smooth_delta, 20)
            
            self.temporal = rs.temporal_filter()

            self.disparity_to_depth = rs.disparity_transform(False)
            
        print("Initialized RealSense Camera\nw: {}, h: {}, depths: {}, frame_rate: {}".format(w, h, depths, frame_rate))

    def __del__(self):
        if not self._pipeline:
            self._pipeline.stop()

    ## Using `with PyRS(...) as pyrs:`:
    # https://stackoverflow.com/questions/1984325/explaining-pythons-enter-and-exit

    def __enter__(self):
        self.start_pipeline()
        print("Started pipeline")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._pipeline:
            self._pipeline.stop()
        print("Closed pipeline")
        
    ## Supporting functions:

    def __initialize_depths_sensor(self):
        '''Don\'t Call'''
        device = self._context.get_device()
        #FIXME: `device.first_depths_sensor` does not work well when multiple RS devices are connected
        self._depths_sensor = device.first_depth_sensor()
        
    def start_pipeline(self):
        '''Always call this function to start the capturing pipeline'''
        self._context = self._pipeline.start(self._config)
        if self.depths_on:
            self.__initialize_depths_sensor()
            self.set_depths_preset(self._preset)
            self._scale = self.get_depth_scale()*1000.0
            
    ## Depths sensor settings:
    
    def get_depths_preset(self):
        '''Return depths sensor\'s preset index'''
        return self._preset
    
    def get_depths_preset_name(self, index):
        '''Return depths sensor\'s preset name form index'''
        return self._depths_sensor.get_option_value_description(rs.option.visual_preset, index)
    
    def get_depth_scale(self):
        '''Get depth scale'''
        return self._depths_sensor.get_depth_scale()
    
    def get_depths_visual_preset_max_range(self):
        '''Returns the depths sensor's visual preset range in Int'''
        assert self.depths_on, 'Error: Depths Sensor was not enabled at initialization (turn `depths` to `True`)'
        return self._depths_sensor.get_option_range(rs.option.visual_preset).max

    def set_depths_preset(self, index):
        '''Sets the depths sensor preset'''
        # http://intelrealsense.github.io/librealsense/doxygen/rs__option_8h.html#a8b9c011f705cfab20c7eaaa7a26040e2
        assert self._preset <= self.get_depths_visual_preset_max_range(), "Error: Desired preset exceeds range"
        self._depths_sensor.set_option(rs.option.visual_preset, index)

    ## Frames:

    def update_frames(self):
        '''Updates frames to pipeline (same as calling `_pipeline.wait_for_frames()`)'''
        frames = self._pipeline.wait_for_frames()

        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        self._color_image = np.asanyarray(color_frame.get_data())

        if self.depths_on:
            self.depths_frame = frames.get_depth_frame()
            #self.depths_frame = self._filter(self.depths_frame)
            # self._depths_image = depths_frame.get_data()
            # print(type(self._depths_image))
            self._depths_image = np.asanyarray(self.depths_frame.get_data())

            # change scale to millimeters
            self._depths_image = self._scale * self._depths_image.astype(np.float64)
            self._depths_image = cv2.resize(self._depths_image.astype(np.uint16), (self._color_image.shape[1], self._color_image.shape[0]), interpolation=self.interpolation)

    def get_color_image(self):
        '''Returns color image as Numpy array'''
        return self._color_image
    
    def get_depths_frame(self):
        '''Returns depth image as Numpy array (in meters)'''
        return self._depths_image

    def get_colorized_depths_frame(self):
        '''Colorize the depth data for rendering'''
        colorized_depth = np.asanyarray(self.colorizer.colorize(self.depths_frame).get_data())
        return cv2.resize(colorized_depth, (self._color_image.shape[1], self._color_image.shape[0]), interpolation=self.interpolation) 

    def _filter(self, frame):
        '''Create filter'''
        frame = self.decimation.process(frame)
        frame = self.depths_to_disparity.process(frame)
        frame = self.spatial.process(frame)
        frame = self.temporal.process(frame)
        frame = self.disparity_to_depth.process(frame)
        return frame

    ## Intrinsics (camera details)

    def get_intrinsic(self, as_json=False, path=None):
        '''Return Camera Intrinsic as json'''
        assert self._context, "Has not started pipeline yet"
        if self.intrinsic is None:
            print("Getting Intrinsics for Camera...")
            profile = self._context.get_stream(rs.stream.color)
            self.intrinsic = profile.as_video_stream_profile().get_intrinsics()
        intrinsic = self._intrinsic2dict(self.intrinsic)  # dict
        if as_json:
            assert path, "You must add path for saving intrinsics"
            intrinsic_as_json = json.dumps(intrinsic)
            with open(os.path.join(path, 'realsense_intrinsic.json'), 'w') as f:
                    intrinsic_as_json = json.dump(intrinsic, f, sort_keys=False,
                                                                indent=4,
                                                                ensure_ascii=False)
            return intrinsic_as_json
        return intrinsic

    def _intrinsic2dict(self, i):
        mat = [i.fx, 0, 0, 0, i.fy, 0, i.ppx, i.ppy, 1]
        return {'width': i.width, 'height': i.height, 'intrinsic_matrix': mat}


if __name__ == '__main__':

    height = 480
    width = 640

    rgb_name = "rgb.png"
    depth_name = "depth.png"

    with PyRS(h=height, w=width) as pyrs:

        print('Modes:')
        print('\tSave RGB and Depths:\tp')
        print('\tChange preset:\tc')
        print('\tSave Intrinsic:\ti')
        print('\tExit:\tq')

        preset = pyrs.get_depths_preset()
        preset_name = pyrs.get_depths_preset_name(preset)
        print('Preset: ', pyrs.get_depths_preset_name(preset))
        print("Intrinsics: ", pyrs.get_intrinsic())

        while True:
            # Wait for a coherent pair of frames: depth and color
            pyrs.update_frames()

            # Get images as numpy arrays
            color_image = pyrs.get_color_image()
            depths_image = pyrs.get_depths_frame()
            colorized_depths = pyrs.get_colorized_depths_frame()

            # Stack both images horizontally
            images = np.hstack((color_image, colorized_depths))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(images, preset_name, (60,80), font, 4,(255,255,255),2, cv2.LINE_AA)

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)

            if key == ord('q'):
                # end OpenCV loop
                break
            elif key == ord('p'):
                # save rgb and depths
                cv2.imwrite(rgb_name, color_image)
                cv2.imwrite(depth_name, depths_image)
            elif key == ord('c'):
                # change preset
                preset = preset + 1
                max_ = pyrs.get_depths_visual_preset_max_range()
                preset = preset % max_
                pyrs.set_depths_preset(preset)
                preset_name = pyrs.get_depths_preset_name(preset)
            elif key == ord('i'):
                # save intrinsics
                intrinsic_as_json = pyrs.get_intrinsic(True, "./")
                
