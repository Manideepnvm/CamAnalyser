import cv2
from deepface import DeepFace
import mediapipe as mp

# Initialize MediaPipe for face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Function to analyze frame
def analyze_frame(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Analyze age, gender, and expression
            face = frame[y:y + h, x:x + w]
            analysis = DeepFace.analyze(face, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            
            # Display results
            age = analysis[0]['age']
            gender = analysis[0]['gender']
            emotion = analysis[0]['dominant_emotion']
            
            # Clear label arrangement
            cv2.putText(frame, f"Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"Age: {age}", (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Get head target
            head_target = get_head_target(x, y, w, h)
            cv2.putText(frame, f"Head Target: {head_target}", (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Function to get head target
def get_head_target(x, y, w, h):
    # Calculate the center of the head based on bounding box
    head_x = x + w // 2
    head_y = y + h // 4  # Approximate head position
    return (head_x, head_y)

# Main function to run webcam
def main():
    cap = cv2.VideoCapture(0)  # Access the webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        analyze_frame(frame)
        
        cv2.imshow('Webcam', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
