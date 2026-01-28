import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = 'test3.mp4'
OUTPUT_VIDEO = 'out2.mp4'

# Strictly tighter thresholds
HIT_ZONE_X = 15
HIT_ZONE_Y = 10
PROXIMITY_LOCK = 80  # Ball must be within this distance to "engage" a cone pass

print("⏳ Loading Model...")
model = YOLO('yolov8m-worldv2.pt')
model.set_classes(["soccer ball", "traffic cone", "orange circle"])

cap = cv2.VideoCapture(VIDEO_PATH)
w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
if fps == 0: fps = 30
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

# --- TRACKING STATES ---
# cone_memory[id] = {"rel_pos": "above/below", "passed_fwd": False, "passed_bwd": False}
cone_memory = {}
dirty_cones = {}

fwd_count = 0
bwd_count = 0
hits = 0
hit_cooldown = 0

print("▶️ Processing with Proximity Lock...")

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

        for box, track_id, cls in zip(boxes, ids, class_ids):
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            if model.names[cls] == "soccer ball" and cx > 10:
                ball_pos = (cx, cy)
            elif model.names[cls] in ["traffic cone", "orange circle"]:
                frame_cones.append({'id': track_id, 'pos': (cx, cy)})

    if ball_pos:
        bx, by = ball_pos

        for cone in frame_cones:
            cid = cone['id']
            cx, cy = cone['pos']

            # Initialize memory for new cones
            if cid not in cone_memory:
                cone_memory[cid] = {"rel_pos": "above" if by < cy else "below"}

            # 1. DISTANCE CHECK (The Proximity Lock)
            dist = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)

            # Only process logic if ball is close to this specific cone
            if dist < PROXIMITY_LOCK:
                # HIT DETECTION
                if abs(bx - cx) < HIT_ZONE_X and abs(by - cy) < HIT_ZONE_Y:
                    dirty_cones[cid] = True
                    if hit_cooldown == 0:
                        hits += 1
                        hit_cooldown = 15
                    cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 2)

                # PASS DETECTION
                current_rel_pos = "above" if by < cy else "below"
                prev_rel_pos = cone_memory[cid]["rel_pos"]

                if prev_rel_pos == "below" and current_rel_pos == "above":
                    # CROSSING FORWARD
                    if not dirty_cones.get(cid, False):
                        fwd_count += 1
                    dirty_cones[cid] = False  # Reset dirty flag after pass

                elif prev_rel_pos == "above" and current_rel_pos == "below":
                    # CROSSING BACKWARD
                    if not dirty_cones.get(cid, False):
                        bwd_count += 1
                    dirty_cones[cid] = False

                cone_memory[cid]["rel_pos"] = current_rel_pos

                # Visual Indicator of 'Active' cone
                cv2.circle(frame, (cx, cy), 25, (255, 0, 255), 1)

            # Draw status
            color = (0, 255, 0) if not dirty_cones.get(cid, False) else (0, 0, 255)
            cv2.circle(frame, (cx, cy), 6, color, -1)

    if hit_cooldown > 0: hit_cooldown -= 1

    # HUD (No denominator, just counts)
    cv2.rectangle(frame, (10, 10), (280, 140), (0, 0, 0), -1)
    cv2.putText(frame, f"Forward: {fwd_count}", (20, 45), 0, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Backward: {bwd_count}", (20, 85), 0, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Hits: {hits}", (20, 125), 0, 0.7, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Directional Tracker", cv2.resize(frame, (1024, 600)))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()