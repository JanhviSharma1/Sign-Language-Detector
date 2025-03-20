import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Constants
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Detect hands
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the cropping area does not go out of bounds
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(img.shape[1], x + w + offset), min(img.shape[0], y + h + offset)

        imgCrop = img[y1:y2, x1:x2]  # Crop the hand region

        # Create a blank white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Resize cropped image while maintaining aspect ratio
        aspectRatio = h / w

        if aspectRatio > 1:
            # If height is greater than width
            newH = imgSize
            newW = int(w * (imgSize / h))
        else:
            # If width is greater than height
            newW = imgSize
            newH = int(h * (imgSize / w))

        imgResize = cv2.resize(imgCrop, (newW, newH))

        # Centering the resized hand inside imgWhite
        xOffset = (imgSize - newW) // 2
        yOffset = (imgSize - newH) // 2
        imgWhite[yOffset:yOffset + newH, xOffset:xOffset + newW] = imgResize

        # Display the images
        cv2.imshow("Webcam", img)
        cv2.imshow("Cropped Hand", imgCrop)
        cv2.imshow("White Image", imgWhite)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
