import cv2
import mediapipe as mp
import pyautogui
import math

def main():
    # --- Initialization ---
    cap = cv2.VideoCapture(0)
    
    # Screen size for mouse mapping
    screen_w, screen_h = pyautogui.size()
    
    # MediaPipe Hands setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    
    # Smoothing variables
    plocX, plocY = 0, 0
    smoothening = 5
    
    print("Anti-Gravity Mouse started. Press ESC to exit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break
            
        # Flip the image horizontally for a natural (mirror) feel
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # 1. Draw hand landmarks on screen
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                
                # 2. Get Landmark coordinates (ID 8: Index Tip, ID 4: Thumb Tip)
                x1 = int(handLms.landmark[8].x * w)
                y1 = int(handLms.landmark[8].y * h)
                x2 = int(handLms.landmark[4].x * w)
                y2 = int(handLms.landmark[4].y * h)
                
                # 3. Draw circles at fingertips for visual feedback
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
                
                # 4. Map coordinates to screen resolution
                mouse_x = screen_w * handLms.landmark[8].x
                mouse_y = screen_h * handLms.landmark[8].y
                
                # 5. Smooth mouse movement
                clocX = plocX + (mouse_x - plocX) / smoothening
                clocY = plocY + (mouse_y - plocY) / smoothening
                
                # 6. Move the cursor
                pyautogui.moveTo(clocX, clocY)
                plocX, plocY = clocX, clocY
                
                # 7. Click Detection (Pinch Gesture)
                # Calculate Euclidean distance between thumb and index finger
                distance = math.hypot(x2 - x1, y2 - y1)
                
                if distance < 35:  # Threshold for pinch
                    pyautogui.click()
                    cv2.putText(img, "CLICK", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Overlay UI Title
        cv2.putText(img, "Anti-Gravity Gesture Mouse", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow("Hand Mouse View", img)
        
        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Anti-Gravity Mouse stopped.")

if __name__ == "__main__":
    main()