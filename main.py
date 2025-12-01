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
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
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

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])


    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frame=video_frames[0],player_detections=tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team_id = team_assigner.get_player_team(frame=video_frames[frame_num],player_bbox=track['bbox'],player_id=player_id)

            tracks['players'][frame_num][player_id]['team_id'] = team_id
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team_id]

    # Assign player ball possession
    player_ball_assigner = PlayerBallAssigner()
    
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player_id = player_ball_assigner.assign_ball_to_player(player_track,ball_bbox)

        if assigned_player_id != -1:
            tracks['players'][frame_num][assigned_player_id]['has_ball'] = True
    
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