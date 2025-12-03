from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self,frames):
        batch_size = 5
        detections = []
        for i in range(0,len(frames),batch_size):
            batch_detections = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += batch_detections
        
        return detections

    def add_object_position_to_tracks(self,tracks):
        for object,object_tracks in tracks.items():
            for frame_num,track in enumerate(object_tracks):
                for track_id,track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    track_info['position'] = position

    def get_object_tracks(self,frames,read_from_stub=False,stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,"rb") as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        tracks = {
            "players" : [],
            "referees" : [],
            "ball" : []
        }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inv = {class_name : idx for idx,class_name in class_names.items()}
            # print(class_names)

            # Convert to Supervision Detection format
            supervision_detections = sv.Detections.from_ultralytics(detection)

            # Convert Goalkeeper class to player class
            for idx,class_id in enumerate(supervision_detections.class_id):
                if class_names[class_id] == "goalkeeper":
                    supervision_detections.class_id[idx] = class_names_inv["player"]
            
            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(supervision_detections)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == class_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox" : bbox}
                
                if cls_id == class_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox" : bbox}
            for frame_detection in supervision_detections:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == class_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox" : bbox}

            # print(tracks)

        if stub_path is not None:
            with open(stub_path,"wb") as f:
                pickle.dump(tracks,f)
        return tracks

    def interpolate_ball_positions(self,ball_tracks):
        ball_positions = [x.get(1,{}).get("bbox",[]) for x in ball_tracks]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate Missing Values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:{"bbox" : x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def draw_annotations(self, video_frames, tracks, video_writer,team_ball_control,camera_movement_per_frame):
        # output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get('team_color',(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame,player['bbox'],(0,0,255))
            
            # Draw referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # Draw team ball control panel
            frame = self.draw_team_ball_control_panel(frame,frame_num,team_ball_control)

            # Draw camera movement
            frame = self.draw_camera_movement(frame,camera_movement_per_frame,frame_num)

            # Write the frame directly to disk
            video_writer.write(frame)
            
            if frame_num % 100 == 0:
                print(f"Processing frame {frame_num}/{len(video_frames)}")
        
        return

    def draw_ellipse(self,frame,bbox,color,track_id=None):
        x1,y1,x2,y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])

        x_center = int((x1 + x2) / 2)
        width = int(x2 - x1)            

        cv2.ellipse(
            img=frame,
            center=(x_center,y2),
            axes=(width,int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235, 
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        rect_width, rect_height = 40, 20
        x1_rect = int(x_center - rect_width // 2)
        x2_rect = int(x_center + rect_width // 2)
        y1_rect = int((y2 - rect_height//2) + 15)
        y2_rect = int((y2 + rect_height//2) + 15)

        if track_id is not None:
            cv2.rectangle(
                img=frame,
                pt1=(x1_rect,y1_rect),
                pt2=(x2_rect,y2_rect),
                color=color,
                thickness=cv2.FILLED
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                img=frame,
                text=f"{track_id}",
                org=(x1_text,y1_rect + 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0,0,0),
                thickness=2
            )
        
        return frame
    
    def draw_traingle(self,frame,bbox,color):
        x1,y1,x2,y2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        x_center = int((x1 + x2) / 2)

        triangle_points = np.array([
            (x_center, y1),
            [x_center-10,y1-20],
            [x_center+10,y1-20]
        ])

        cv2.drawContours(
            image=frame,
            contours=[triangle_points],
            contourIdx=0,
            color=color,
            thickness=cv2.FILLED
        )  
        cv2.drawContours(
            image=frame,
            contours=[triangle_points],
            contourIdx=0,
            color=(0,0,0),
            thickness=2
        )  
        return frame
        
    # def draw_annotations(self,video_frames,tracks):
    #     output_video_frames = []
    #     for frame_num, frame in enumerate(video_frames):
    #         frame = frame.copy()

    #         player_dict = tracks["players"][frame_num]
    #         referee_dict = tracks["referees"][frame_num]
    #         ball_dict = tracks["ball"][frame_num]

    #         # Darw players
    #         for track_id, player in player_dict.items():
    #             frame = self.draw_ellipse(frame,player["bbox"],(0,0,255),track_id)
            
    #         output_video_frames.append(frame)
    #     return output_video_frames

    def draw_team_ball_control_panel(self,frame,frame_num,team_ball_control):
        # Draw opaque rectangle
        overlay = frame.copy()
        cv2.rectangle(
            img=overlay,
            pt1=(1350,850),
            pt2=(1900,970),
            color=(255,255,255),
            thickness=-1
        )
        alpha = 0.4
        cv2.addWeighted(
            src1=overlay,
            alpha=alpha,
            src2=frame,
            beta=1-alpha,
            gamma=0,
            dst=frame
        )

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the no. of times each team has the ball
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(
            img=frame,
            text=f"Team 1 ball control: {team_1*100:.2f}%",
            org=(1400,900),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale = 1,
            color=(0,0,0),
            thickness=3
        )
        cv2.putText(
            img=frame,
            text=f"Team 2 ball control: {team_2*100:.2f}%",
            org=(1400,950),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale = 1,
            color=(0,0,0),
            thickness=3
        )

        return frame

    def draw_camera_movement(self,frame,camera_movement_per_frame,frame_num):
        overlay = frame.copy()
        cv2.rectangle(
            img=overlay,
            pt1=(0,0),
            pt2=(500,100),
            color=(255,255,255),
            thickness=-1
        )
        alpha = 0.6
        cv2.addWeighted(
            src1=overlay,
            alpha=alpha,
            src2=frame,
            beta=1-alpha,
            gamma=0,
            dst=frame
        )
        x_movement,y_movement = camera_movement_per_frame[frame_num]
        cv2.putText(
            img=frame,
            text=f"Camera X Movement: {x_movement:.2f}",
            org=(10,30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0,0,0),
            thickness=3
        )
        cv2.putText(
            img=frame,
            text=f"Camera Y Movement: {y_movement:.2f}",
            org=(10,60),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0,0,0),
            thickness=3
        )        
        return frame