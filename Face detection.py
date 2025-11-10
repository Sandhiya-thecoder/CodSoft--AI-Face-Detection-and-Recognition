import cv2
import numpy as np
import os
from cv2 import face

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a face recognizer using LBPH
recognizer = face.LBPHFaceRecognizer_create()

# Path for the trained face image
trained_face_path = "trained_face.png"

# Flag to check if training has been done
trained = False

# Flag to stop the program
stop_program = False

# Function to train the recognizer on the saved face
def train_on_saved_face():
    global trained
    if os.path.exists(trained_face_path):
        # Load the saved face image
        img = cv2.imread(trained_face_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize for consistency (same as in recognition)
            img = cv2.resize(img, (200, 200))
            # Train the recognizer
            recognizer.train([img], np.array([0]))  # Label 0 for the trained face
            recognizer.save("trainer.yml")
            trained = True
            print("Training completed on the saved face.")
        else:
            print("Could not load the saved face image.")
    else:
        print("No trained face image found.")

# Main program loop
while True and not stop_program:
    # Initialize video capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        break
    
    print("Webcam opened. Press 't' to train/save face, 'q' to quit this mode, 's' to stop the program.")
    
    while True and not stop_program:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=7,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Face Detection", frame)
        
        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("t"):  # Train/Save face
            if len(faces) > 0:
                # Take the first detected face
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                # Save the face
                cv2.imwrite(trained_face_path, face_roi)
                print(f"Face saved as '{trained_face_path}'.")
                # Train the recognizer
                train_on_saved_face()
                # Close the window
                cv2.destroyAllWindows()
                video_capture.release()
                break  # Exit inner loop to wait for next action
            else:
                print("No face detected to save.")
        elif key == ord("q"):  # Quit this mode (but if trained, proceed to recognition)
            cv2.destroyAllWindows()
            video_capture.release()
            if trained:
                # Reopen webcam for recognition
                video_capture = cv2.VideoCapture(0)
                if not video_capture.isOpened():
                    print("Error: Could not reopen webcam.")
                    break
                
                print("Webcam reopened for recognition. Detecting face... Press 's' to stop.")
                
                # Load the trained model
                if os.path.exists("trainer.yml"):
                    recognizer.read("trainer.yml")
                
                while True and not stop_program:
                    ret, frame = video_capture.read()
                    if not ret:
                        print("Error: Could not read frame.")
                        break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,
                        minNeighbors=7,
                        minSize=(50, 50),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    if len(faces) > 0:
                        # Take the first detected face for comparison
                        x, y, w, h = faces[0]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (200, 200))
                        
                        # Predict
                        id_, confidence = recognizer.predict(face_roi)
                        if id_ == 0 and confidence < 70:  # Match threshold
                            result_text = "Recognized"
                            color = (0, 255, 0)  # Green for recognized
                            print("Recognized")
                        else:
                            result_text = "Different face"
                            color = (0, 0, 255)  # Red for different
                            print("Different face")
                        
                        # Display the result on the frame
                        cv2.putText(frame, result_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                        # Show the frame with the text
                        cv2.imshow("Recognition Result", frame)
                        
                        # Wait for a key press to continue or exit
                        key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for key
                        if key == ord("s"):  # Stop key
                            stop_program = True
                            break
                        
                        # Close after displaying result
                        cv2.destroyAllWindows()
                        video_capture.release()
                        break  # Exit recognition loop
                    else:
                        # Display frame while waiting for face
                        cv2.imshow("Recognition - Detect Face", frame)
                        if cv2.waitKey(1) & 0xFF == ord("s"):  # Stop key
                            stop_program = True
                            break
                
                break  # Exit outer loop after recognition
            else:
                print("No trained face available. Train first by pressing 't'.")
                break
        elif key == ord("s"):  # Stop the program
            stop_program = True
            break
    
    # If we broke out of the inner loop, check if we should continue or exit
    if stop_program or not trained or key == ord("q"):
        break

# Final cleanup
cv2.destroyAllWindows()
print("Program ended.")
