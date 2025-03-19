import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector with one max hand
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Detect hands
    hands, img = detector.findHands(img)  # Finds the hands and draws on the image

    # Display the image
    cv2.imshow("Image", img)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
