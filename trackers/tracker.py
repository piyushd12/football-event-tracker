from ultralytics import YOLO
import supervision as sv
import pickle
import os

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self,frames):
        batch_size = 20
        detections = []
        for i in range(0,len(frames),batch_size):
            batch_detections = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += batch_detections
        
        return detections

    def get_object_tracks(self,frames,read_from_stub=False,stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,"rb") as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)

        tracks = {
            "player" : [],
            "referee" : [],
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

            tracks["player"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == class_names_inv["player"]:
                    tracks["player"][frame_num][track_id] = {"bbox" : bbox}
                
                if cls_id == class_names_inv["referee"]:
                    tracks["referee"][frame_num][track_id] = {"bbox" : bbox}

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

