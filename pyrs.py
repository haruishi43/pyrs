import os
import json

import cv2
import pyrealsense2 as rs
import numpy as np


class PyRS:
    def __init__(self, w=640, h=480, depth=True, frame_rate=30):
        '''
        Initializing the Python RealSense Control Flow:
        w: Int (default = 640, can also be 1280) 
        h: Int (default = 480, can also be 720)
        depth: Bool (default = True)
        frame_rate: Int (default = 30)

        RGB and Depth formats are: bgr8, z16

        Note: In this class, variables should not be directly changed.
        '''
        self.width = w
        self.height = h
        self.depth_on = depth
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, frame_rate)
        self._intrinsic = None

        if depth:
            self.align = rs.align(rs.stream.color)
            self._preset = 0
            # Presets (for D415):
            # 0: Custom
            # 1: Default
            # 2: Hand 
            # 3: High Accuracy
            # 4: High Density
            # 5: Medium Density

            # depth interpolation
            self.interpolation = cv2.INTER_NEAREST  # use nearest neighbor
            # self.interpolation = cv2.INTER_LINEAR  # linear
            # self.interpolation = cv2.INTER_CUBIC  # cubic

            # beautify depth image for viewing
            self._config.enable_stream(rs.stream.depth, w, h, rs.format.z16, frame_rate)
            self.colorizer = rs.colorizer()

            # initialize filters
            self.decimation = rs.decimation_filter()
            self.decimation.set_option(rs.option.filter_magnitude, 4)

            self.depth_to_disparity = rs.disparity_transform(True)

            self.spatial = rs.spatial_filter()
            self.spatial.set_option(rs.option.filter_magnitude, 5)
            self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial.set_option(rs.option.filter_smooth_delta, 20)
            
            self.temporal = rs.temporal_filter()

            self.disparity_to_depth = rs.disparity_transform(False)
            
        print("Initialized RealSense Camera\nw: {}, h: {}, depth: {}, frame_rate: {}".format(w, h, depth, frame_rate))

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

    def __initialize_depth_sensor(self):
        '''
        Initialize depth sensor
        Don\'t Call this function anywhere else!!!
        '''
        device = self._context.get_device()
        #FIXME: `device.first_depth_sensor` does not work well when multiple RS devices are connected
        #TODO: input device id when multiple realsense devices are connected
        self._depth_sensor = device.first_depth_sensor()
        
    def start_pipeline(self):
        '''Always call this function to start the capturing pipeline'''
        self._context = self._pipeline.start(self._config)
        if self.depth_on:
            self.__initialize_depth_sensor()

            # set preset
            self.set_depth_preset(self._preset)

            # set scale of depth
            self._scale = self._get_depth_scale()
            
    ## depth sensor settings:
    
    def get_depth_preset(self):
        '''Return depth sensor\'s preset index'''
        return self._preset
    
    def get_depth_preset_name(self, index):
        '''Return depth sensor\'s preset name form index'''
        return self._depth_sensor.get_option_value_description(rs.option.visual_preset, index)
    
    def get_depth_visual_preset_max_range(self):
        '''Returns the depth sensor's visual preset range in Int'''
        assert self.depth_on, 'Error: depth Sensor was not enabled at initialization (turn `depth` to `True`)'
        return self._depth_sensor.get_option_range(rs.option.visual_preset).max

    def set_depth_preset(self, index):
        '''Sets the depth sensor preset'''
        # http://intelrealsense.github.io/librealsense/doxygen/rs__option_8h.html#a8b9c011f705cfab20c7eaaa7a26040e2
        assert self._preset <= self.get_depth_visual_preset_max_range(), "Error: Desired preset exceeds range"
        self._depth_sensor.set_option(rs.option.visual_preset, index)

    def _get_depth_scale(self):
        '''Get depth scale in millimeter'''
        return self._depth_sensor.get_depth_scale() * 1000.0

    ## Frames:

    def update_frames(self):
        '''Updates frames to pipeline (same as calling `_pipeline.wait_for_frames()`)'''
        # wait for frames
        frames = self._pipeline.wait_for_frames()

        if self.depth_on:
            # align depth to rgb
            frames = self.align.process(frames)

            # get depth frame
            self.depth_frame = frames.get_depth_frame()

            # filter depth frame
            self.depth_frame = self._filter(self.depth_frame)

            # depth data to numpy array
            self._depth_image = np.asanyarray(self.depth_frame.get_data())

            # change scale (to uint16)
            self._depth_image = (self._scale * self._depth_image.astype(np.float64)).astype(np.uint16)

            # resize depth image to match color image size
            self._depth_image = cv2.resize(self._depth_image, (self.width, self.height), interpolation=self.interpolation)
        
        # get color frame
        color_frame = frames.get_color_frame()

        # color data to numpy array
        self._color_image = np.asanyarray(color_frame.get_data())


    def get_color_image(self):
        '''Returns color image as Numpy array'''
        return self._color_image
    
    def get_depth_frame(self):
        '''Returns depth image as Numpy array'''
        return self._depth_image

    def get_colorized_depth_frame(self):
        '''Colorize the depth data for rendering'''
        colorized_depth = np.asanyarray(self.colorizer.colorize(self.depth_frame).get_data())
        return cv2.resize(colorized_depth, (self.width, self.height), interpolation=self.interpolation) 

    def _filter(self, frame):
        '''Filter depth frame'''
        frame = self.decimation.process(frame)
        frame = self.depth_to_disparity.process(frame)
        frame = self.spatial.process(frame)
        frame = self.temporal.process(frame)
        frame = self.disparity_to_depth.process(frame)
        return frame

    ## Intrinsics (camera details)

    def get_intrinsic(self, as_json=False, path=None):
        '''Return Camera Intrinsic as json'''
        assert self._context, "Has not started pipeline yet"

        # only obtain intrinsics once...
        if self._intrinsic is None:
            print("Getting Intrinsics for Camera...")
            
            # Depth image is aligned to color, therefore get the intrinsics for color sensor
            profile = self._context.get_stream(rs.stream.color)
            self._intrinsic = profile.as_video_stream_profile().get_intrinsics()

        # convert to dict
        intrinsic = self._intrinsic2dict(self._intrinsic)

        # save as json in `path`
        if as_json:
            assert path, "You must add path for saving intrinsics"
            intrinsic_as_json = json.dumps(intrinsic)
            with open(os.path.join(path, 'realsense_intrinsic.json'), 'w') as f:
                    intrinsic_as_json = json.dump(intrinsic, f, sort_keys=False,
                                                                indent=4,
                                                                ensure_ascii=False)
            return intrinsic_as_json

        # just return dict
        return intrinsic

    def _intrinsic2dict(self, i):
        '''Convert intrinsic data to dict (readable in Open3D)'''
        mat = [i.fx, 0, 0, 0, i.fy, 0, i.ppx, i.ppy, 1]
        return {'width': i.width, 'height': i.height, 'intrinsic_matrix': mat}


if __name__ == '__main__':

    height = 720
    width = 1280

    rgb_name = "rgb.png"
    depth_name = "depth.png"

    with PyRS(h=height, w=width) as pyrs:

        print('Modes:')
        print('\tSave RGB and depth:\tp')
        print('\tChange preset:\tc')
        print('\tSave Intrinsic:\ti')
        print('\tExit:\tq')

        preset = pyrs.get_depth_preset()
        preset_name = pyrs.get_depth_preset_name(preset)
        print('Preset: ', pyrs.get_depth_preset_name(preset))
        print("Intrinsics: ", pyrs.get_intrinsic())

        while True:
            # Wait for a coherent pair of frames: depth and color
            pyrs.update_frames()

            # Get images as numpy arrays
            color_image = pyrs.get_color_image()
            depth_image = pyrs.get_depth_frame()
            colorized_depth = pyrs.get_colorized_depth_frame()

            # Stack both images horizontally
            images = np.hstack((color_image, colorized_depth))

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(images, preset_name, (60,80), font, 4, (255,255,255), 2, cv2.LINE_AA)

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)

            if key == ord('q'):
                # end OpenCV loop
                break
            elif key == ord('p'):
                # save rgb and depth
                cv2.imwrite(rgb_name, color_image)
                cv2.imwrite(depth_name, depth_image)
            elif key == ord('c'):
                # change preset
                preset = preset + 1
                max_ = pyrs.get_depth_visual_preset_max_range()
                preset = preset % max_
                pyrs.set_depth_preset(preset)
                preset_name = pyrs.get_depth_preset_name(preset)
            elif key == ord('i'):
                # save intrinsics
                intrinsic_as_json = pyrs.get_intrinsic(True, "./")
                
