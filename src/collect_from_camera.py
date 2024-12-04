import cv2
import mediapipe as mp
import csv
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# CSV file for saving data
output_csv = "emotion_data.csv"

# Define emotions
emotions = ["neutral", "happy", "sad", "angry", "lying"]  # Add more emotions as needed

# Start webcam and collect data
cap = cv2.VideoCapture(0)

current_emotion = None
print(f"Available emotions: {emotions}")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect landmarks
    results = face_mesh.process(rgb_frame)

    # Draw landmarks and capture data
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Collect landmarks as a flat vector
            landmarks = [coord for landmark in face_landmarks.landmark for coord in (landmark.x, landmark.y, landmark.z)]

            # Save landmarks to file if an emotion is selected
            if current_emotion:
                with open(output_csv, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([current_emotion] + landmarks)

    # Display the frame
    cv2.putText(frame, f"Emotion: {current_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Emotion Data Collection", frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Quit
        break
    elif key in [ord(e[0]) for e in emotions]:  # Switch emotion (e.g., 'n' for neutral)
        current_emotion = emotions[[ord(e[0]) for e in emotions].index(key)]

cap.release()
cv2.destroyAllWindows()
