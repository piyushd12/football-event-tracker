from ultralytics import YOLO

model = YOLO("/media/piyush/DATA/football-event-tracker/yolov8n.pt") 

results = model.predict(source="/media/piyush/DATA/football-event-tracker/input_videos/Screencast from 2025-11-27 19-09-46.mp4", save=True)
print(results)
print('=========================')
print(results[0])
print('=========================')
for box in results[0].boxes:
    print(box)