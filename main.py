
from matplotlib.pylab import f
from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator

def main():
    #read video
    video_frames = read_video("input_videos/08fd33_4.mp4")
    #init tracker

    tracker = Tracker("models/yolov8x_best_finetunned.pt")

    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub=True,
                                        stub_path="stubs/track_stubs.pkl")
    

    #get object positions
    tracker.add_position_to_tracks(tracks)
    
    # camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path="stubs/camera_movement_stub.pkl")
    # adjust object positions
    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # view transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    
    #speed and distance estimator of players
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_distance_to_tracks(tracks)

    # assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    # iterate for each player in each frame and assign a team
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # assign ball to player (ball aquisition)
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_ball_assigner.assign_ball_to_player(
                                        player_track,
                                        ball_bbox
                                        )
        
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    #save cropped image of player
    """for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]

        # crop bbox from frame
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        cv2.imwrite(f"output_videos/cropped_imgage.jpg", cropped_image)
        break"""

                                       
    # draw output
    # draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    #output_video_frames = tracker.draw_annotations(video_frames, tracks)
    # save video
    # save_video(video_frames, "output_videos/output_video.mp4")
    #draw camera movement 
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    #draw speed and distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    save_video(output_video_frames, "output_videos/output_video.mp4")

if __name__ == "__main__":
    main()