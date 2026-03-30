import cv2
import mediapipe as mp
import pyautogui
import math

# Camera
cap = cv2.VideoCapture(0)

# Screen size
screen_w, screen_h = pyautogui.size()

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Smooth movement
plocX, plocY = 0, 0
smoothening = 5

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    h, w, c = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # Draw hand
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Index finger tip (8)
            x1 = int(handLms.landmark[8].x * w)
            y1 = int(handLms.landmark[8].y * h)

            # Thumb tip (4)
            x2 = int(handLms.landmark[4].x * w)
            y2 = int(handLms.landmark[4].y * h)

            # Draw points
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)

            # Convert to screen coordinates
            mouse_x = screen_w * x1 / w
            mouse_y = screen_h * y1 / h

            # Smooth movement
            clocX = plocX + (mouse_x - plocX) / smoothening
            clocY = plocY + (mouse_y - plocY) / smoothening

            pyautogui.moveTo(clocX, clocY)
            plocX, plocY = clocX, clocY

            # Click detection (pinch)
            distance = math.hypot(x2 - x1, y2 - y1)

            if distance < 30:
                pyautogui.click()
                cv2.putText(img, "CLICK", (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # Title
    cv2.putText(img, "Anti-Gravity Mouse", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()