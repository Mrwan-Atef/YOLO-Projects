import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = 'test2.mp4'  # Replace with your video path
OUTPUT_VIDEO = 'output_slalom.mp4'

# Sensitivity settings
HIT_THRESHOLD_X = 25  # Horizontal distance (pixels) to count a hit (Strict)
HIT_THRESHOLD_Y = 15  # Vertical distance (pixels) - smaller because of perspective
CONE_PASS_OFFSET = 20  # How many pixels "past" the cone counts as a pass

# Load Model
model = YOLO('yolov8m-worldv2.pt')
# Added "yellow flat marker" and "disc cone" for the second image you showed
model.set_classes(["person", "soccer ball", "traffic cone", "orange circle", "yellow flat marker", "disc cone"])

cap = cv2.VideoCapture(VIDEO_PATH)

# Video Writer Setup
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# State Variables
cones_passed_count = 0
total_cones = 0
cone_hits = 0
passed_cone_ids = set()  # Keep track of which cones we've already passed
hit_cooldown = 0

print("Processing Slalom Drill...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. TRACKING
    results = model.track(frame, persist=True, conf=0.15, verbose=False)

    player_pos = None
    ball_pos = None
    cones = []  # List of (x, y, id)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().numpy()
        names = model.names

        for box, track_id, cls in zip(boxes, ids, class_ids):
            name = names[cls]
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            feet_y = int(y2)  # Bottom of the box

            if name == "person":
                player_pos = (center_x, feet_y)
            elif name == "soccer ball":
                ball_pos = (center_x, center_y)
            elif name in ["traffic cone", "orange circle", "yellow flat marker", "disc cone"]:
                # Store cone with its ID so we track specific cones
                cones.append({'id': track_id, 'pos': (center_x, center_y), 'hit': False})

    # 2. DRILL LOGIC (SLALOM)
    if cones:
        # Sort cones by Y (Vertical drill: Bottom -> Top)
        # We assume the player starts at the bottom (High Y) and moves to top (Low Y)
        # If your video is Left->Right, change this to sort by X.
        sorted_cones = sorted(cones, key=lambda c: c['pos'][1], reverse=False)  # Top (small Y) to Bottom (big Y)
        total_cones = len(sorted_cones)

        # Draw the "Path" (Line connecting cones)
        for i in range(len(sorted_cones) - 1):
            cv2.line(frame, sorted_cones[i]['pos'], sorted_cones[i + 1]['pos'], (200, 200, 0), 1)

    # 3. ANALYSIS
    if ball_pos:
        ball_x, ball_y = ball_pos

        for cone in cones:
            cx, cy = cone['pos']
            c_id = cone['id']

            # A. HIT DETECTION (Elliptical/Box Logic)
            # We check X and Y separately.
            # The ball must be VERY close horizontally (X), but can be slightly further vertically (Y) due to perspective.
            dx = abs(ball_x - cx)
            dy = abs(ball_y - cy)

            if dx < HIT_THRESHOLD_X and dy < HIT_THRESHOLD_Y:
                if hit_cooldown == 0:
                    cone_hits += 1
                    hit_cooldown = 20  # Wait 20 frames before counting another hit

                # Visual Feedback for Hit
                cv2.circle(frame, (cx, cy), 25, (0, 0, 255), 3)
                cv2.putText(frame, "HIT", (cx - 20, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # B. PASS COUNTER (Dribbling Past)
            # Assuming player moves UP (Y decreases):
            # If ball is ABOVE the cone (ball_y < cone_y) AND we haven't counted this cone yet
            if ball_y < (cy - CONE_PASS_OFFSET):
                if c_id not in passed_cone_ids:
                    passed_cone_ids.add(c_id)
                    cones_passed_count += 1

            # Draw Cones (Green if passed, Orange if not)
            color = (0, 255, 0) if c_id in passed_cone_ids else (0, 165, 255)
            cv2.circle(frame, (cx, cy), 5, color, -1)

    # 4. DRAWING DASHBOARD
    cv2.rectangle(frame, (20, 20), (350, 120), (0, 0, 0), -1)

    # Progress Bar Text
    progress_text = f"Cones Passed: {cones_passed_count} / {total_cones}"
    cv2.putText(frame, progress_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hit Counter
    cv2.putText(frame, f"Mistakes (Hits): {cone_hits}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Cooldown timer decrement
    if hit_cooldown > 0: hit_cooldown -= 1

    # Save and Show
    out.write(frame)

    # Resize for your screen preview
    display_frame = cv2.resize(frame, (1024, 600))
    cv2.imshow("Slalom Analysis", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()