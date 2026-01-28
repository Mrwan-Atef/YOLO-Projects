import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = 'test3.mp4'
OUTPUT_VIDEO = 'out1.mp4'

# Detection Zones
HIT_ZONE_X = 15  # Horizontal width of cone impact area
HIT_ZONE_Y = 10  # Vertical height of cone impact area
PASS_MARGIN = 5  # Buffer to prevent flickering at the line

print("⏳ Loading Model...")
model = YOLO('yolov8m-worldv2.pt')
# Restricted classes: No flat markers included
model.set_classes(["soccer ball", "traffic cone", "orange circle"])

cap = cv2.VideoCapture(VIDEO_PATH)
w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
if fps == 0: fps = 30
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

# --- TRACKING STATES ---
# cone_states[id] = "below" or "above"
cone_states = {}
# dirty_cones[id] = True if ball touched this cone during this crossing
dirty_cones = {}

forward_passes = 0
backward_passes = 0
total_hits = 0
hit_cooldown = 0

print("▶️ Processing Directional Slalom...")

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model.track(frame, persist=True, conf=0.25, verbose=False)

    ball_pos = None
    frame_cones = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().numpy()
        names = model.names

        for box, track_id, cls in zip(boxes, ids, class_ids):
            name = names[cls]
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

            if name == "soccer ball" and cx > 10:
                ball_pos = (cx, cy)
            elif name in ["traffic cone", "orange circle"]:
                frame_cones.append({'id': track_id, 'pos': (cx, cy)})

    if ball_pos:
        bx, by = ball_pos

        for cone in frame_cones:
            cid = cone['id']
            cx, cy = cone['pos']

            # 1. HIT DETECTION
            # If ball is inside the cone's hit box, mark this cone as "Dirty"
            if abs(bx - cx) < HIT_ZONE_X and abs(by - cy) < HIT_ZONE_Y:
                dirty_cones[cid] = True
                if hit_cooldown == 0:
                    total_hits += 1
                    hit_cooldown = 15
                cv2.circle(frame, (cx, cy), 30, (0, 0, 255), 2)

            # 2. PASS LOGIC (State Transition)
            # Determine if ball is currently 'Above' (lower Y) or 'Below' (higher Y) the cone
            current_rel_pos = "above" if by < cy else "below"

            # Check if state changed from previous frame
            if cid in cone_states:
                prev_rel_pos = cone_states[cid]

                if prev_rel_pos == "below" and current_rel_pos == "above":
                    # FORWARD PASS: Ball moved from near camera to far
                    if not dirty_cones.get(cid, False):
                        forward_passes += 1
                    # Reset dirty status after a crossing is completed
                    dirty_cones[cid] = False

                elif prev_rel_pos == "above" and current_rel_pos == "below":
                    # BACKWARD PASS: Ball moved from far to near camera
                    if not dirty_cones.get(cid, False):
                        backward_passes += 1
                    dirty_cones[cid] = False

            # Update memory
            cone_states[cid] = current_rel_pos

            # Visual Markers
            color = (0, 255, 0) if not dirty_cones.get(cid, False) else (0, 0, 255)
            cv2.circle(frame, (cx, cy), 6, color, -1)

    if hit_cooldown > 0: hit_cooldown -= 1

    # --- HUD ---
    cv2.rectangle(frame, (10, 10), (320, 150), (0, 0, 0), -1)
    cv2.putText(frame, f"Forward Passes: {forward_passes}", (20, 45), 0, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Backward Passes: {backward_passes}", (20, 85), 0, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Total Hits: {total_hits}", (20, 125), 0, 0.7, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Clean Pass Tracker", cv2.resize(frame, (1024, 600)))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()