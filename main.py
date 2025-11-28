from utils import read_video, save_video
from trackers import Tracker
# import time

def main():
    # Read Video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # ini = time.time()
    # Initialize Tracker
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="stubs/tracks_stub.pkl")
    
    # print(time.time() - ini)
    # save video
    # save_video(video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()