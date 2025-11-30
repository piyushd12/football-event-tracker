import cv2
from utils import read_video
from trackers import Tracker
import time

def main():
    ini = time.time()
    input_video_path = "input_videos/08fd33_4.mp4"
    video_name = input_video_path.split("/")[-1].split(".")[0]
    video_frames = read_video(input_video_path)
    n1 = time.time()
    print(f"Time to read video: {n1 - ini} seconds")

    # Initialize Tracker
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=f"stubs/tracks_stub_{video_name}.pkl")
    n2 = time.time()
    print(f"Time to get object tracks: {n2 - n1} seconds")

    for track_id,player in tracks['players'][0].items():
        bbox = player['bbox']
        cropped_img = video_frames[0][int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        cv2.imwrite(f"output_videos/cropped_player_{track_id}.jpg", cropped_img)
        break

if __name__ == "__main__":
    main()