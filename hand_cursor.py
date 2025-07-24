import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import time
screen_w, screen_h = pyautogui.size()
click_state=False
click_state_middle=False
i=0
def get_point(lm, w, h):
    return [lm.x * w, lm.y * h]

def get_angle(mcp,pip,tip,w,h):
    a = get_point(mcp, w, h)
    b = get_point(pip, w, h)
    c = get_point(tip, w, h)
    vec1 = np.array(a) - np.array(b)
    vec2 = np.array(c) - np.array(b)
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Function to draw landmarks
def draw_landmarks_on_image(rgb_image, hand_landmarks_list):
    annotated_image = np.copy(rgb_image)

    for hand_landmarks in hand_landmarks_list:
        for landmark in hand_landmarks:
            x_px = int(landmark.x * annotated_image.shape[1])
            y_px = int(landmark.y * annotated_image.shape[0])
            cv2.circle(annotated_image, (x_px, y_px), 5, (0, 255, 0), -1)
    return annotated_image
    
#Function for cursor movement
def cursor_move(rgb_image, hand_landmarks_list):
    annotated_image = np.copy(rgb_image)
    if hand_landmarks_list:
    #for hand_landmarks in hand_landmarks_list:
        index_tip = hand_landmarks_list[0][8]  # Index fingertip is landmark 8
        #x_px = int(index_tip.x * annotated_image.shape[1])
        #y_px = int(index_tip.y * annotated_image.shape[0])
        #cv2.circle(annotated_image, (x_px, y_px), 10, (0, 0, 255), -1)  # Red circle

        # Map to screen coordinates (inverted x because webcam is mirrored)
        x = abs(screen_w-int(index_tip.x * screen_w))
        y = int(index_tip.y * screen_h)
        pyautogui.moveTo(x, y)



# Global variable to hold latest result
latest_result = None

# Callback function
def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = (result, output_image)

# Set up HandLandmarker with LIVE_STREAM and callback
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=result_callback
)
detector = vision.HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Get timestamp in ms
    timestamp = int(time.time() * 1000)

    # Send frame to the detector
    detector.detect_async(mp_image, timestamp)

    # If there's a recent result, draw landmarks
    if latest_result:
        result, _ = latest_result
        if(result.hand_landmarks):
            cursor_move(rgb_frame, result.hand_landmarks)
            thumb_angle=get_angle(result.hand_landmarks[0][1],result.hand_landmarks[0][2],result.hand_landmarks[0][4],rgb_frame.shape[1],rgb_frame.shape[0])
            middle_angle=get_angle(result.hand_landmarks[0][9],result.hand_landmarks[0][10],result.hand_landmarks[0][12],rgb_frame.shape[1],rgb_frame.shape[0])
            
            #If thumb is bent => left click
            if thumb_angle < 140 and not click_state:
                pyautogui.click()
                click_state = True
                print("click!",i)
                i+=1
            elif thumb_angle >= 140:
                    click_state = False

            #If middle finger is bent => right click
            if middle_angle < 140 and not click_state_middle:
                pyautogui.rightClick()
                click_state_middle = True
                print("click middle!",i)
                i+=1
            elif middle_angle >= 140:
                click_state_middle = False
        annotated_frame = draw_landmarks_on_image(rgb_frame, result.hand_landmarks)
    else:
        annotated_frame = rgb_frame

    # Display the frame
    cv2.imshow("Live Hand Landmarks", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
