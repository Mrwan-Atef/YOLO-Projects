from ultralytics import YOLO
import cv2
import numpy as np
# load yolo model
model = YOLO('yolov8n.pt')
model_world = YOLO('yolov8s-world.pt')
model_world_v2 = YOLO('yolov8s-worldv2.pt')
model_world_v2.set_classes([ "soccer ball" , "person" , "traffic cone" ])

# load the video

video_path='test3.mp4'
cap= cv2.VideoCapture(video_path)

# read frames
ret = True
while ret:
    ret , frame = cap.read()
    # detect and track objects
    results = model_world_v2.track(frame , verbose=True , persist=True , conf=0.1)

    #plot
    frame_ = results[0].plot()
    display_frame = cv2.resize(frame_, (960, 540))
    #visulaize
    cv2.imshow('frame',display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()