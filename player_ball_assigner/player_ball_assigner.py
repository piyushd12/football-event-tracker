import sys
sys.path.append('../')
from utils import  get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self,player_tracks,ball_bbox):
        ball_posisiton = get_center_of_bbox(ball_bbox)

        min_distance = float('inf')
        assigned_player_id = -1

        for player_id,player in player_tracks.items():
            player_bbox = player['bbox']
            left_distance = measure_distance((player_bbox[0],player_bbox[3]),ball_posisiton)
            right_distance = measure_distance((player_bbox[2],player_bbox[3]),ball_posisiton)
            distance = min(left_distance,right_distance)

            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                assigned_player_id = player_id
        
        return assigned_player_id

            