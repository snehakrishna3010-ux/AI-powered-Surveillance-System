import cv2

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imshow("AI Surveillance Camera", frame)

    # press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()