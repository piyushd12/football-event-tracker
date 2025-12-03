import pickle
import cv2
import numpy as np
import os
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator:
    def __init__(self,frame):
        self.min_distance = 5

        first_frame_grayscale = cv2.cvtColor(src=frame,code=cv2.COLOR_BGR2GRAY) 
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,:20] = 1
        mask_features[:,900:1050] = 1

        self.features_params = {
            'maxCorners':100,
            'qualityLevel':0.3,
            'minDistance':3,
            'blockSize':7,
            'mask':mask_features
        }

        self.lk_params = {
            'winSize':(15,15),
            'maxLevel':2,
            'criteria':(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        }


    def add_adjusted_positions_to_tracks(self,tracks,camera_movement_per_frame):
        for object,object_tracks in tracks.items():
            for frame_num,track in enumerate(object_tracks):
                for track_id,track_info in track.items():
                    position = track_info['position']
                    curr_camera_movement = camera_movement_per_frame[frame_num]
                    adjusted_position = (position[0]-curr_camera_movement[0],position[1]-curr_camera_movement[1])
                    track_info['adjusted_position'] = adjusted_position
                

    def get_camera_movement(self,frames,read_from_stub=False,stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                camera_movement = pickle.load(f)
            return camera_movement

        camera_movement = [[0,0]] * len(frames)

        initial_frame_grayscale = cv2.cvtColor(src=frames[0],code=cv2.COLOR_BGR2GRAY)
        initial_features = cv2.goodFeaturesToTrack(
            image=initial_frame_grayscale,
            **self.features_params
        )

        for frame_num in range(1,len(frames)):
            curr_frame_graycale = cv2.cvtColor(src=frames[frame_num],code=cv2.COLOR_BGR2GRAY)
            curr_features,_,_ = cv2.calcOpticalFlowPyrLK(
                prevImg=initial_frame_grayscale,
                nextImg=curr_frame_graycale,
                prevPts=initial_features,
                nextPts=None,
                **self.lk_params
            )

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0
            for i ,(ini,curr) in enumerate(zip(initial_features,curr_features)):
                ini_feature_point = ini.ravel()
                curr_feature_point = curr.ravel()

                curr_distance = measure_distance(ini_feature_point,curr_feature_point)
                if curr_distance > max_distance:
                    max_distance = curr_distance
                    camera_movement_x,camera_movement_y = measure_xy_distance(ini_feature_point,curr_feature_point)
            
            if max_distance > self.min_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                initial_features = curr_features.copy()
                # initial_features = cv2.goodFeaturesToTrack(
                #     image=curr_frame_graycale,
                #     **self.features_params
                # )

            initial_frame_grayscale = curr_frame_graycale.copy()
        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)
                
        return camera_movement