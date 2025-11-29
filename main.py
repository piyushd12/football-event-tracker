# from utils import read_video, save_video
# from trackers import Tracker
# import time

# def main():
#     # Read Video
#     video_frames = read_video("input_videos/08fd33_4.mp4")

#     ini = time.time()
#     # Initialize Tracker
#     tracker = Tracker("models/best.pt")
#     tracks = tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="stubs/tracks_stub.pkl")
    
#     # Draw output annotations
#     # Draw object Tracks
#     output_video_frames = tracker.draw_annotations(video_frames,tracks)

#     now = time.time()
#     print(now - ini)
#     # save video
#     save_video(output_video_frames, "output_videos/output_video.avi")
#     print(time.time() - now)


# if __name__ == "__main__":
#     main()

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
    
    # Setup Output Video Writer
    output_path = f"output_videos/output_video_{video_name}.avi"
    height, width = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 24, (width, height))

    # Pass the writer to the function
    tracker.draw_annotations(video_frames, tracks, out)
    n3 = time.time()
    print(f"Time to draw annotations: {n3 - n2} seconds")
    
    out.release() 
    print("Video saved successfully.")
    n4 = time.time()
    print(f"Time to save video: {n4 - n3} seconds")

if __name__ == "__main__":
    main()