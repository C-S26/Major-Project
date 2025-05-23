import cv2
import os

# Create known_faces directory if not exists
output_dir = "known_faces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ask user for the name ONCE
person_name = input("Enter name for the face to capture: ").strip()
face_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Make a copy of frame for saving clean faces
    original_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Draw rectangle on the *frame* only for display
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Capturing Faces - Press "c" to Capture, "q" to Quit', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Save face from the *original_frame* without rectangles
        for (x, y, w, h) in faces:
            face_img = original_frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (150, 150))

            filename = os.path.join(output_dir, f"{person_name}_{face_count}.jpg")
            cv2.imwrite(filename, face_img)
            print(f"[INFO] Saved: {filename}")
            face_count += 1

    elif key == ord('q'):
        break

# Release webcam
video_capture.release()
cv2.destroyAllWindows()
