import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = 'test3.mp4'
OUTPUT_VIDEO = 'out5.mp4'

# "Pass Distance": How far "past" the line you must go to trigger (prevents flickering)
HYSTERESIS_BUFFER = 10
# "Side Distance": You must be within this many pixels of the cone to count (don't count if running far away)
MAX_SIDE_DIST = 100

print(" Loading Model...")
model = YOLO('yolov8m-worldv2.pt')
model.set_classes(["soccer ball", "traffic cone", "orange circle", "person"])

cap = cv2.VideoCapture(VIDEO_PATH)
w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
if fps == 0: fps = 30
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# --- STATE VARIABLES ---
is_calibrated = False
cone_data = []  # Stores {id, pos, axis_pos}
# cone_states[id] = "BEFORE" or "AFTER"
cone_states = {}

# Global vectors for the course direction
course_dir = np.array([0, 0])
course_normal = np.array([0, 0])

total_passes = 0
total_hits = 0
hit_cooldowns = {}

print("Phase 1: Calibrating Course Geometry...")

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model.track(frame, persist=True, conf=0.25, verbose=False)

    # 1. DATA COLLECTION
    ball_pos = None
    target_type = None
    frame_cones = []

    # Temporary lists
    balls = []
    people = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, class_ids):
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            name = model.names[cls]

            if name == "soccer ball":
                balls.append(np.array([cx, cy]))
            elif name == "person":
                people.append(np.array([cx, cy]))
            elif name in ["traffic cone", "orange circle"]:
                frame_cones.append({'id': track_id, 'pos': np.array([cx, cy])})

    # Prioritize Ball > Person
    if balls:
        ball_pos = balls[0]
        target_type = "BALL"
    elif people:
        ball_pos = people[0]
        target_type = "PERSON"

    # 2. CALIBRATION (Run Once)
    # This mathematically finds the "Direction" of your slalom
    if not is_calibrated and len(frame_cones) >= 4:
        # Sort cones by X to find Start and End
        sorted_cones = sorted(frame_cones, key=lambda c: c['pos'][0])

        start_node = sorted_cones[0]['pos']  # Left-most
        end_node = sorted_cones[-1]['pos']  # Right-most

        # Vector from First to Last Cone
        diff = end_node - start_node
        length = np.linalg.norm(diff)

        if length > 0:
            course_dir = diff / length  # Normalized Direction Vector (Unit Vector)
            # Perpendicular Vector (Rotate 90 degrees) [-y, x]
            course_normal = np.array([-course_dir[1], course_dir[0]])

            # Store Cones
            cone_data = sorted_cones
            is_calibrated = True
            print(f" Calibration Complete. Direction: {course_dir}")

    # 3. GEOMETRIC TRACKING
    if is_calibrated and ball_pos is not None:
        bx, by = ball_pos

        # Visualization: Draw the tracker
        color = (0, 255, 0) if target_type == "BALL" else (255, 0, 0)
        cv2.circle(frame, (bx, by), 8, color, -1)

        for cone in cone_data:
            cid = cone['id']
            c_pos = cone['pos']

            # Vector from Cone to Ball
            vec_cb = ball_pos - c_pos

            # A. PROJECTED DISTANCE (Along the course direction)
            # Negative = Before Cone, Positive = After Cone
            dist_along_course = np.dot(vec_cb, course_dir)

            # B. SIDE DISTANCE (Perpendicular from center line)
            # Are you close enough to the cone to count?
            dist_from_center = abs(np.dot(vec_cb, course_normal))

            # Initialize State
            if cid not in cone_states:
                # Determine where we started (Before or After)
                cone_states[cid] = "BEFORE" if dist_along_course < 0 else "AFTER"

            # --- CROSSING LOGIC ---
            # We only care if "dist_along_course" flips sign AND we are close enough

            if dist_from_center < MAX_SIDE_DIST:
                # Visual Debug: Draw the "Gate" Line
                # We draw a line perpendicular to course direction
                p1 = (c_pos + course_normal * 40).astype(int)
                p2 = (c_pos - course_normal * 40).astype(int)

                # Color code gate: Yellow = Waiting, Green = Passed
                gate_color = (0, 255, 255) if cone_states[cid] == "BEFORE" else (0, 255, 0)
                cv2.line(frame, tuple(p1), tuple(p2), gate_color, 2)

                # Logic: Crossing "Forward" (Before -> After)
                if cone_states[cid] == "BEFORE" and dist_along_course > HYSTERESIS_BUFFER:
                    cone_states[cid] = "AFTER"
                    total_passes += 1
                    print(f" Passed Cone {cid}")

                # Logic: Crossing "Backward" (After -> Before - if doing laps)
                elif cone_states[cid] == "AFTER" and dist_along_course < -HYSTERESIS_BUFFER:
                    cone_states[cid] = "BEFORE"
                    total_passes += 1
                    print(f"Passed Cone {cid} (Return)")

            # Hit Detection (Simple Radius)
            if np.linalg.norm(vec_cb) < 15:
                if hit_cooldowns.get(cid, 0) == 0:
                    total_hits += 1
                    hit_cooldowns[cid] = 30
                    cv2.circle(frame, tuple(c_pos), 20, (0, 0, 255), 3)

        # Update cooldowns
        for k in hit_cooldowns:
            if hit_cooldowns[k] > 0: hit_cooldowns[k] -= 1

    # DASHBOARD
    cv2.rectangle(frame, (10, 10), (220, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"Passes: {total_passes}", (20, 45), 0, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Hits: {total_hits}", (20, 85), 0, 0.8, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Vector Crossing Logic", cv2.resize(frame, (1024, 600)))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()