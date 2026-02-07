import cv2
import os

def extract_all_frames(video_path, output_folder):
    # 1. Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while True:
        # 3. Read the next frame
        ret, frame = cap.read()
        
        # If there are no more frames, break the loop
        if not ret:
            break

        # 4. Save the frame
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        if frame_count % 100 == 0:
            print(f"Saved {frame_count} frames...")
        
        frame_count += 1

    # 5. Clean up
    cap.release()
    print(f"Done! Total frames extracted: {frame_count}")

# Example Usage
extract_all_frames('dribbling-videos\test3.mp4', 'extracted_frames_cones')