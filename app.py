import cv2
import face_recognition
import os
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load known faces
known_face_encodings = []
known_face_names = []

path = "known_faces"

for file in os.listdir(path):
    img = face_recognition.load_image_file(os.path.join(path, file))
    encodings = face_recognition.face_encodings(img)

    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])
        known_face_names.append(file.split(".")[0])

# Create intruder folder
intruder_folder = "intruders"
os.makedirs(intruder_folder, exist_ok=True)

last_intruder_time = 0

# Start camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0,0), fx=0.7, fy=0.7)

    # Run YOLO
    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:

            if int(box.cls[0]) == 0:   # person class

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw person box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size == 0:
                    continue

                rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_crop)
                face_encodings = face_recognition.face_encodings(rgb_crop, face_locations)

                for (top,right,bottom,left),face_encoding in zip(face_locations,face_encodings):

                    face_distances = face_recognition.face_distance(
                        known_face_encodings,
                        face_encoding
                    )

                    name = "Unknown"
                    color = (0,0,255)

                    if len(face_distances) > 0:

                        best_match_index = face_distances.argmin()

                        if face_distances[best_match_index] < 0.6:
                            name = known_face_names[best_match_index]
                            color = (0,255,0)

                    cv2.rectangle(person_crop,(left,top),(right,bottom),color,2)

                    cv2.putText(person_crop,name,(left,top-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

                    # Save intruder image
                    if name == "Unknown":

                        if time.time() - last_intruder_time > 5:

                            filename = f"{intruder_folder}/intruder_{int(time.time())}.jpg"

                            cv2.imwrite(filename, frame)

                            print("⚠ Intruder detected:", filename)

                            last_intruder_time = time.time()

    cv2.imshow("AI Surveillance System", frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()