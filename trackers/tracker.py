
from ultralytics import YOLO
import supervision as sv



class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        #we will use goalkeepers as players (small dataset 631 fine tuned) with trakers and not predict
        self.tracker = sv.ByteTrack()
    def detect_frames(self, frames):
        batch_size = 20
        detected_frames = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i: i + batch_size], conf = 0.1)#conf -> confidence threshold
            detected_frames += detections_batch
            break
        return detected_frames 

    def get_object_tracks(self, frames):
         
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
            tracks["players"].append(detection_with_tracks)
            tracks["referees"].append(detection_with_tracks)
            tracks["ball"].append(detection_with_tracks)
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist() #position of bbox(bounding box) in the detections list
                cls_id= frame_detection[3] #position of class id in the detections list
                track_id = frame_detection[4] #position of track id in the detections list
                
                if cls_id == cls_names_inv["player"]:
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


            print(detection_supervision)