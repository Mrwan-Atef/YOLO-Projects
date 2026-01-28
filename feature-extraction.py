import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = 'test2.mp4'
FPS = 30
PIXELS_PER_METER = 100

# Distances
POSSESSION_DIST_PX = 80  # Ball is "with" the player
CONE_HIT_DIST_PX = 40  # Ball hit the cone
PASS_COMPLETE_OFFSET = 50  # How far past the cones the ball must go to count

# Load Model
model = YOLO('yolov8m-worldv2.pt')
model.set_classes(["person", "soccer ball", "traffic cone", "orange circle"])

cap = cv2.VideoCapture(VIDEO_PATH)

# Variables
pass_count = 0
cone_hits = 0
ball_in_play = False  # Is the ball currently moving away from player?
current_shot_clean = True  # Did this specific shot hit a cone yet?
shot_cooldown = 0  # Prevent double counting one shot

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. TRACKING
    results = model.track(frame, persist=True, conf=0.1, verbose=False)

    player_pos = None
    ball_pos = None
    cones = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().numpy()
        names = model.names

        for box, cls in zip(boxes, class_ids):
            name = names[cls]
            x1, y1, x2, y2 = box
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            feet_pos = (int((x1 + x2) / 2), int(y2))

            if name == "person":
                player_pos = feet_pos
            elif name == "soccer ball":
                ball_pos = center
            elif name in ["traffic cone", "orange circle", "cone"]:
                cones.append(center)

    # 2. DEFINE THE "TARGET LINE"
    # We find the cone that is furthest away (smallest Y value usually)
    # or the average Y of all cones to define where the "gate" is.
    target_line_y = 0
    if cones:
        # Assuming cones are arranged horizontally, we take the average Y
        # If cones are arranged vertically (ladder), take the furthest one (min Y)
        cone_ys = [c[1] for c in cones]
        target_line_y = int(np.mean(cone_ys))

        # Draw the virtual finish line
        cv2.line(frame, (0, target_line_y), (frame.shape[1], target_line_y), (255, 0, 0), 2)
        cv2.putText(frame, "TARGET LINE", (10, target_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 3. GAME LOGIC
    if ball_pos and player_pos and target_line_y > 0:

        # Calculate Distance Player <-> Ball
        dist_to_player = np.linalg.norm(np.array(player_pos) - np.array(ball_pos))

        # A. STATE: PLAYER HAS BALL (RESET)
        if dist_to_player < POSSESSION_DIST_PX:
            ball_in_play = False
            current_shot_clean = True  # Reset for new shot

        # B. STATE: BALL IS KICKED (IN PLAY)
        elif dist_to_player > POSSESSION_DIST_PX and not ball_in_play:
            if shot_cooldown == 0:
                ball_in_play = True  # Shot started!

        # C. CHECK FOR HITS
        if ball_in_play:
            # Check collisions with ANY cone
            for cone in cones:
                dist_cone = np.linalg.norm(np.array(ball_pos) - np.array(cone))
                if dist_cone < CONE_HIT_DIST_PX:
                    # HIT DETECTED!
                    if current_shot_clean:  # Only register the first hit of this shot
                        cone_hits += 1
                        current_shot_clean = False  # Mark this shot as "dirty"
                        shot_cooldown = 30  # Wait a bit

                    cv2.circle(frame, cone, 30, (0, 0, 255), -1)
                    cv2.putText(frame, "HIT!", cone, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # D. CHECK FOR SUCCESS (CROSSING THE LINE)
            # Logic: If ball Y is ABOVE the line (smaller value) and clean
            # Note: In OpenCV, Y=0 is the TOP of the screen.
            # If player is at bottom shooting UP, we check if ball_y < target_line_y

            # Check if ball passed the line
            if ball_pos[1] < target_line_y:
                if current_shot_clean and shot_cooldown == 0:
                    pass_count += 1
                    shot_cooldown = 60  # Don't count the same pass twice

                    # VISUAL SUCCESS
                    cv2.putText(frame, "CLEAN PASS!", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    # Cooldown timer
    if shot_cooldown > 0:
        shot_cooldown -= 1

    # 4. DASHBOARD
    cv2.rectangle(frame, (20, 20), (300, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Clean Passes: {pass_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Cone Hits: {cone_hits}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Single Player Drill", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()