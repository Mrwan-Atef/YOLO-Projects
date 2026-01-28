import cv2
import numpy as np
from ultralytics import YOLO

# --- REFINED CONFIGURATION ---
VIDEO_PATH = 'test3.mp4'
OUTPUT_VIDEO = 'out4.mp4'

# Adjusted for smaller cone sizes and wider turns
HIT_ZONE_X = 15
HIT_ZONE_Y = 10
PROXIMITY_LIMIT = 120  # [FIX] Increased from 85 to catch wide turns
GATE_OFFSET = 5  # [FIX] Smaller gap so even shallow movements count

print(" Loading Model...")
model = YOLO('yolov8m-worldv2.pt')
model.set_classes(["soccer ball", "traffic cone", "orange circle"])

cap = cv2.VideoCapture(VIDEO_PATH)
w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
if fps == 0: fps = 30
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

# --- MEMORY ---
cone_memory = {}
fwd_count = 0
bwd_count = 0
hits = 0
hit_cooldown = 0

print(" Processing: High Sensitivity Mode Active...")

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model.track(frame, persist=True, conf=0.25, verbose=True)

    ball_pos = None
    frame_cones = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, class_ids):
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            if model.names[cls] == "soccer ball":
                ball_pos = (cx, cy)
            elif model.names[cls] in ["traffic cone", "orange circle"]:
                frame_cones.append({'id': track_id, 'pos': (cx, cy)})

    if ball_pos and frame_cones:
        bx, by = ball_pos

        # 1. EXCLUSIVE TARGETING (Focus only on the nearest cone)
        dists = [np.sqrt((bx - c['pos'][0]) ** 2 + (by - c['pos'][1]) ** 2) for c in frame_cones]
        min_idx = np.argmin(dists)
        active_cone = frame_cones[min_idx] if dists[min_idx] < PROXIMITY_LIMIT else None

        for cone in frame_cones:
            cid = cone['id']
            cx, cy = cone['pos']

            # Initial Setup for each cone
            if cid not in cone_memory:
                # Auto-detect starting position: is the ball currently above or below?
                initial_state = "above" if by < cy else "below"
                cone_memory[cid] = {"state": initial_state, "dirty": False}

            # 2. HIT DETECTION
            if abs(bx - cx) < HIT_ZONE_X and abs(by - cy) < HIT_ZONE_Y:
                cone_memory[cid]["dirty"] = True
                if hit_cooldown == 0:
                    hits += 1
                    hit_cooldown = 20
                cv2.circle(frame, (cx, cy), 25, (0, 0, 255), 3)  #

            # 3. PASS LOGIC (Only for active cone)
            if active_cone and cid == active_cone['id']:
                # Draw the purple circle for the target
                cv2.circle(frame, (cx, cy), 35, (255, 0, 255), 2)

                # Check for crossing
                if by > (cy + GATE_OFFSET):
                    # DECISIVELY BELOW
                    if cone_memory[cid]["state"] == "above":
                        # Ball crossed from Above to Below (BACKWARD)
                        if not cone_memory[cid]["dirty"]:
                            bwd_count += 1
                            cv2.putText(frame, "BACKWARD PASS!", (bx, by - 20), 0, 0.6, (255, 255, 0), 2)
                        cone_memory[cid]["dirty"] = False
                    cone_memory[cid]["state"] = "below"

                elif by < (cy - GATE_OFFSET):
                    # DECISIVELY ABOVE
                    if cone_memory[cid]["state"] == "below":
                        # Ball crossed from Below to Above (FORWARD)
                        if not cone_memory[cid]["dirty"]:
                            fwd_count += 1
                            cv2.putText(frame, "FORWARD PASS!", (bx, by - 20), 0, 0.6, (0, 255, 0), 2)
                        cone_memory[cid]["dirty"] = False
                    cone_memory[cid]["state"] = "above"

            # Draw static status dots
            dot_color = (0, 0, 255) if cone_memory[cid]["dirty"] else (0, 255, 0)
            cv2.circle(frame, (cx, cy), 6, dot_color, -1)

    if hit_cooldown > 0: hit_cooldown -= 1

    # DASHBOARD
    cv2.rectangle(frame, (10, 10), (300, 140), (0, 0, 0), -1)
    cv2.putText(frame, f"Forward: {fwd_count}", (20, 45), 0, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Backward: {bwd_count}", (20, 85), 0, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Hits: {hits}", (20, 125), 0, 0.7, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Sensitivity Fixed", cv2.resize(frame, (1024, 600)))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()