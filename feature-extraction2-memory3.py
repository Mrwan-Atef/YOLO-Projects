import cv2
import numpy as np
from ultralytics import YOLO


VIDEO_PATH = 'test3.mp4'
OUTPUT_VIDEO = 'out3.mp4'

# Thresholds
HIT_ZONE_X = 15
HIT_ZONE_Y = 10
PROXIMITY_LIMIT = 70  # Only the closest cone within this range activates
GATE_OFFSET = 15  # The "buffer" zone to prevent flicker (Hysteresis)
print("Loading Model")
model = YOLO('yolov8m-worldv2.pt')
model.set_classes(["soccer ball", "traffic cone", "orange circle"])

cap = cv2.VideoCapture(VIDEO_PATH)
w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
if fps == 0: fps = 30
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

# --- MEMORY ---
# cone_memory[id] = {"state": "waiting/entered_top/entered_bottom", "dirty": False}
cone_memory = {}
fwd_count = 0
bwd_count = 0
hits = 0
hit_cooldown = 0

print("Processing with Gate Logic & Exclusive Targeting")

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
            if model.names[cls] == "soccer ball" and cx > 10:
                ball_pos = (cx, cy)
            elif model.names[cls] in ["traffic cone", "orange circle"]:
                frame_cones.append({'id': track_id, 'pos': (cx, cy)})

    if ball_pos and frame_cones:
        bx, by = ball_pos

        # 1. FIND THE SINGLE CLOSEST CONE (Exclusive Targeting)
        # This prevents circles from appearing on two cones at once.
        distances = [np.sqrt((bx - c['pos'][0]) ** 2 + (by - c['pos'][1]) ** 2) for c in frame_cones]
        min_dist = min(distances)
        closest_cone = frame_cones[distances.index(min_dist)] if min_dist < PROXIMITY_LIMIT else None

        for cone in frame_cones:
            cid = cone['id']
            cx, cy = cone['pos']

            if cid not in cone_memory:
                cone_memory[cid] = {"state": "neutral", "dirty": False}

            # 2. HIT DETECTION (Always active)
            if abs(bx - cx) < HIT_ZONE_X and abs(by - cy) < HIT_ZONE_Y:
                cone_memory[cid]["dirty"] = True
                if hit_cooldown == 0:
                    hits += 1
                    hit_cooldown = 20
                cv2.circle(frame, (cx, cy), 25, (0, 0, 255), 2)

            # 3. GATE LOGIC (Only for the active/closest cone)
            if closest_cone and cid == closest_cone['id']:
                # Draw "Active" circle
                cv2.circle(frame, (cx, cy), 30, (255, 0, 255), 1)

                # State Machine
                # Top Gate is cy - GATE_OFFSET | Bottom Gate is cy + GATE_OFFSET
                if by > (cy + GATE_OFFSET):
                    # Ball is decisively BELOW the cone
                    if cone_memory[cid]["state"] == "entered_top":
                        # We just came from the top -> BACKWARD PASS COMPLETE
                        if not cone_memory[cid]["dirty"]:
                            bwd_count += 1
                        cone_memory[cid]["dirty"] = False  # Reset for next pass
                    cone_memory[cid]["state"] = "entered_bottom"

                elif by < (cy - GATE_OFFSET):
                    # Ball is decisively ABOVE the cone
                    if cone_memory[cid]["state"] == "entered_bottom":
                        # We just came from the bottom -> FORWARD PASS COMPLETE
                        if not cone_memory[cid]["dirty"]:
                            fwd_count += 1
                        cone_memory[cid]["dirty"] = False
                    cone_memory[cid]["state"] = "entered_top"

            # Draw static status
            dot_color = (0, 0, 255) if cone_memory[cid]["dirty"] else (0, 255, 0)
            cv2.circle(frame, (cx, cy), 5, dot_color, -1)

    if hit_cooldown > 0: hit_cooldown -= 1

    # HUD
    cv2.rectangle(frame, (15, 15), (280, 145), (0, 0, 0), -1)
    cv2.putText(frame, f"Forward: {fwd_count}", (30, 50), 0, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Backward: {bwd_count}", (30, 90), 0, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Hits: {hits}", (30, 130), 0, 0.7, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Gate Logic Tracker", cv2.resize(frame, (1024, 600)))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()