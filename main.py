import cv2
from ultralytics import YOLO
import numpy as np
from video_utility import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner


def main():
    # Read the video
    video_frames = read_video("input_videos/soccer_clip.mp4")

    # Initialize Tracker
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pk1')

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                  track['bbox'],
                                                  player_id)
            tracks['players'][frame_num][player_id]["team"] = team
            tracks['players'][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # Assign ball to player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            if assigned_player in tracks["players"][frame_num]:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                print(f"Warning: Assigned player {assigned_player} not found in frame {frame_num}")
                if team_ball_control:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    team_ball_control.append(None)
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(None)

    team_ball_control = np.array(team_ball_control)

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Save the video
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == '__main__':
    main()