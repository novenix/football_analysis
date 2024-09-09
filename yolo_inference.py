from ultralytics import YOLO

model = YOLO('models/yolov8x_best_finetunned.pt')  # Load model

results = model.predict('input_videos/08fd33_4.mp4',save = True)
print(results[0])
print('#####################')

#bounding boxes
for box in results[0].boxes:  # type: ignore
    print(box)


"""exmaple of bounding box 
This is for person
cls: tensor([0.])
conf: tensor([0.8621])
data: tensor([[533.6715, 686.3803, 579.3327, 784.5319,   0.8621,   0.0000]])
id: None
is_track: False
orig_shape: (1080, 1920)
shape: torch.Size([1, 6])
xywh: tensor([[556.5021, 735.4561,  45.6612,  98.1516]]) xy whidth and height, center of the box
xywhn: tensor([[0.2898, 0.6810, 0.0238, 0.0909]]) normalized xywh
xyxy: tensor([[533.6715, 686.3803, 579.3327, 784.5319]])
xyxyn: tensor([[0.2780, 0.6355, 0.3017, 0.7264]])
ultralytics.engine.results.Boxes object with attributes:

"""
