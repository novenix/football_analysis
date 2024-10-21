
from turtle import position
from scipy import interpolate
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
import os
import pickle
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv
import sys

sys.path.append('../')


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        #we will use goalkeepers as players (small dataset 631 fine tuned) with trakers and not predict
        self.tracker = sv.ByteTrack()



    #interpolate ball positions
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detected_frames = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i: i + batch_size], conf = 0.1)#conf -> confidence threshold
            detected_frames += detections_batch
            #break
        return detected_frames 

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]["position"] = position

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        stub_exists = os.path.exists(stub_path) if stub_path is not None else False
        print(str(read_from_stub)+ " " + (stub_path if stub_path is not None else "None")+ " " + str(stub_exists))
       
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print("Reading tracks from stub file")
            # check if the stub file exists and read the tracks from it
            # avoid to run the model again
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)
        #improve output of the model
        
        tracks={
            "players": [],
            "referees": [],
            "ball": []
        }

        #we will use goalkeepers as players (small dataset 631 fine tuned) with trakers and not predict
        #overrade goalkepers with players
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            print(cls_names)
            #cls_names_inv class names inverted
            cls_names_inv = {v: k for k, v in cls_names.items()}
            #convert detection for supervision format detection
            detection_supervision = sv.Detections.from_ultralytics(detection)
            #convert goalkeepers to players object
            for object_index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_index] = cls_names_inv["player"]

            #track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            """tracks["players"].append(detection_with_tracks)
            tracks["referees"].append(detection_with_tracks)
            tracks["ball"].append(detection_with_tracks)"""
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()  # position of bbox(bounding box) in the detections list
                cls_id = frame_detection[3]  # position of class id in the detections list
                track_id = frame_detection[4]  # position of track id in the detections list
                print("cls_id: " + str(cls_id))
                print("bbox: " + str(bbox))
                print("track_id: " + str(track_id))
                print("frame_num: " + str(frame_num))
                print("bbox"+str(bbox))
                if cls_id == cls_names_inv["player"]:
                    """ print(type(tracks["players"]))
                    if bbox.ndim == 1:
                        bbox = bbox.reshape(1, -1)
                    print(type(tracks["players"][frame_num]))
                    print(type(tracks["players"][frame_num][track_id]))
                    print(type({"bbox": bbox}))
                    print((tracks["players"][frame_num][track_id]))
                    print("bbox" + bbox)"""
                   
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                #to improve the detection we only will suppose is only one ball in the pitch

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                #no track id because we know is numer one in dataset 
                track_id = 1
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][track_id] = {"bbox": bbox}

        if stub_path is not None:
            # save tracks to a file
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

            # print(detection_supervision)
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int( bbox[3])


        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45, #angle of circle
            endAngle=235,
            color=color,
            lineType=cv2.LINE_4,
            thickness=2,
        )
        #rectangle 
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) +15
        y2_rect = (y2 + rectangle_height // 2) +15
        
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int( y1_rect)),
                          (int(x2_rect),int( y2_rect)),
                          color,
                          cv2.FILLED,  # rectangle filled
                          )
            x1_text = x1_rect + 12
            if track_id > 99:

                x1_text = -10  # to center the text when number is pretty big

            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                        )

        return frame
    
    def daw_triangle(self, frame, bbox, color):
        y = int(bbox[3])
        x, _ = get_center_of_bbox(bbox)
        
        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
            ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED) #triangle filled
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)#triangle border

        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # draw a semi-transparent rectangle to show the team that has the ball control
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970),(255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        #calculate the percentage of time that a team has the ball control
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        
        #get the number of time each ime have the ball
        team_1_num_frames = team_ball_control_till_frame[
                        team_ball_control_till_frame == 1
                        ].shape[0]
        team_2_num_frames = team_ball_control_till_frame[
                        team_ball_control_till_frame == 2
                        ].shape[0]
        #make statistics
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)
        
        cv2.putText(frame, f"TEAM 1 BALL CONTROL: {team_1*100: .2f}%", (1400, 900), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"TEAM 2 BALL CONTROL: {team_2*100: .2f}%", (1400, 950), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

        return frame
    

    def draw_annotations(self, video_frames, tracks, team_ball_control):
    #def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            # copy the frame to draw the annotations on it
            frame = frame.copy()
            #extract players, referees and ball from tracks
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            #draw players

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame,
                                          player["bbox"],
                                          color,
                                          track_id)
                if player.get("has_ball", False):
                    frame = self.daw_triangle(frame,
                                              player["bbox"],
                                              (0, 0, 255))
            #draw referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, 
                                          referee["bbox"],
                                          (0, 255, 255))
            #draw ball
            for track_id, ball in ball_dict.items():
                frame = self.daw_triangle(frame,
                                          ball["bbox"],
                                          (0, 255, 0))
                
            #draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num,
                                                team_ball_control)

            # output_video_frames.append(frame_with_annotations)
            output_video_frames.append(frame)

        return output_video_frames

