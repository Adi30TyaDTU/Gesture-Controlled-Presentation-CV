import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Variables for gesture control
width, height = 1200, 860
folderPath = "Presentation_1"  # Replace with your own folder path or make it configurable

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# List of Presentation Images (replace with your own logic to list images)
pathImages = sorted(os.listdir(folderPath), key=len)

# Variables for gesture control
imgNumber = 0
hs, ws = int(120 * 1.5), int(213 * 1)  # width and height of small image
gestureThreshold = 420
buttonPressed = False
buttonCounter = 0
buttonDelay = 30
annotations = [[]]
annotationNumber = 0
annotationStart = False
smoothFactor = 0.5  # Increase this value for more smoothness
drawingEnabled = False  # Flag to check if drawing is enabled

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Initialize previous index finger position
prev_x, prev_y = 0, 0
smoothed_indexFinger = (0, 0)

while True:
    # Import Images (replace with your own logic to load images)
    success, img = cap.read()
    img = cv2.flip(img, 1)  # 0 is Vertical and 1 is Horizontal
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Resize the slide to fit within the display window
    imgCurrent = cv2.resize(imgCurrent, (width, height - 100))

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (1900, gestureThreshold), (0, 255, 0), 10)

    if hands and not buttonPressed:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        handType = hand['type']  # Determine if the hand is 'Left' or 'Right'
        lmList = hand['lmList']

        # Constraints values for easy drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        indexFinger = (xVal, yVal)

        # Apply exponential moving average for smoothing
        smoothed_indexFinger = (
            int(smoothFactor * indexFinger[0] + (1 - smoothFactor) * prev_x),
            int(smoothFactor * indexFinger[1] + (1 - smoothFactor) * prev_y)
        )
        prev_x, prev_y = smoothed_indexFinger

        if cy <= gestureThreshold:
            # Gesture-1: Left
            if (handType == 'Left' and fingers == [1, 0, 0, 0, 0]) or (handType == 'Right' and fingers == [0, 0, 0, 0, 1]):
                if imgNumber > 0:
                    annotations = [[]]
                    annotationNumber = 0
                    annotationStart = False
                    buttonPressed = True
                    imgNumber -= 1
            # Gesture-2: Right
            if (handType == 'Right' and fingers == [1, 0, 0, 0, 0]) or (handType == 'Left' and fingers == [0, 0, 0, 0, 1]):
                if imgNumber < len(pathImages) - 1:
                    annotations = [[]]
                    annotationNumber = 0
                    annotationStart = False
                    buttonPressed = True
                    imgNumber += 1

        # Gesture-3 : Pointer
        if fingers == [0, 1, 1, 0, 0] or fingers == [0, 1, 0, 0, 0]:
            cv2.circle(imgCurrent, smoothed_indexFinger, 12, (255, 0, 0), cv2.FILLED)

        # Gesture-4 : Draw
        if fingers == [0, 1, 0, 0, 0] and drawingEnabled:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, smoothed_indexFinger, 12, (255, 0, 0), cv2.FILLED)
            annotations[annotationNumber].append(smoothed_indexFinger)

        else:
            annotationStart = False

        # Gesture-5: Erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True
    else:
        annotationStart = False

    # Button Press Iterations
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonPressed = False
            buttonCounter = 0

    # Draw annotations
    for i in range(len(annotations)):
        for j in range(1, len(annotations[i])):
            cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 2)  # Adjust thickness here

    # Adding webcam images on to slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall

    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('w'):
        drawingEnabled = not drawingEnabled  # Toggle drawing enabled
    elif key == ord('c'):
        annotations = [[]]  # Clear all drawings
        annotationNumber = 0

cv2.destroyAllWindows()
cap.release()
