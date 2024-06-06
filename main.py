import cv2
import mediapipe as mp
import numpy as np

# Load the YOLO model and coco class labels
net = cv2.dnn.readNet("darknet/yolov3.weights", "darknet/cfg/yolov3.cfg")
classes = []
with open("darknet/data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up the video capture device
cap = cv2.VideoCapture(0)



# Create a MediaPipe Hand Landmark model object
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# Create a MediaPipe DrawingSpec object for drawing the hand landmarks and connections
mp_drawing = mp.solutions.drawing_utils
landmark_color = (0, 255, 0) # BGR
connection_color = (255, 0, 0)
drawing_spec = mp_drawing.DrawingSpec(landmark_color, thickness=5, circle_radius=2)
connection_spec = mp_drawing.DrawingSpec(connection_color, thickness=5, circle_radius=2)

# Set the parameters for detecting the wave gesture
threshold = 0.20
max_frames_without_wave = 5
frames_without_wave = 2
last_wave = False

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Convert the frame to RGB and process it with the MediaPipe Hand Landmark model
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    # Draw the hand landmarks and connections on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the landmarks for the thumb, index, and middle fingers
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Calculate the distances between the thumb, index, and middle fingers
            dist1 = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
            dist2 = ((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)**0.5

            # Determine if the distances indicate a wave gesture
            if dist1 < threshold and dist2 < threshold:
                # If the gesture has been sustained for multiple frames, mark it as a wave
                frames_without_wave = 0
                if not last_wave:
                    last_wave = True
                    print("Hand shaking motion detected. No more coffee needed.")
            else:
                # If the gesture has not been sustained for multiple frames, reset the wave detector
                frames_without_wave += 1
                if frames_without_wave >= max_frames_without_wave:
                    last_wave = False

            # Draw the hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, drawing_spec, connection_spec)
            
            # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Process the output detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Filter for person class (class_id=0)
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Count the number of people
    num_people = len(boxes)

    # Draw bounding boxes and labels on the frame
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "Person"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), font, 0.5, (0, 255, 0), 2)
            
    # Display the resulting image
    cv2.putText(frame, "People: {}".format(num_people), (10, 30), font, 0.7, (0, 0, 255), 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Person counter and Hand Landmark Tracking', frame)
    # Display the frame with bounding boxes and person count

    #cv2.imshow("Person Counter", frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(30)
