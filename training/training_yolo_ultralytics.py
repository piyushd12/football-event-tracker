import os
import ultralytics
from roboflow import Roboflow
from dotenv import load_dotenv
import shutil

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov5")

# print(dataset.location)
# convert into desired ultralytics format
shutil.move(f'{dataset.location}/test', f'{dataset.location}/{os.path.basename(dataset.location)}/test')
shutil.move(f'{dataset.location}/train', f'{dataset.location}/{os.path.basename(dataset.location)}/train')
shutil.move(f'{dataset.location}/valid', f'{dataset.location}/{os.path.basename(dataset.location)}/valid')

model = ultralytics.YOLO('yolov5xu.pt')
model.train(data=f'{dataset.location}/data.yaml', model='yolov5xu.pt', epochs=100, imgsz=640, cache=True, patience=10, pretrained=True, val=True)