import cv2
import mediapipe as mp

from config import *

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.6,
    max_num_hands=2
)


def get_fingertip_landmarks(fingertip_landmark) -> list:
    """convert from Decimal position to pixels and return it"""
    landmark_x = fingertip_landmark.x
    landmark_y = fingertip_landmark.y
    landmark_z = fingertip_landmark.z

    cx = int(IMG_SHAPE_X * landmark_x)
    cy = int(IMG_SHAPE_Y * landmark_y)

    return landmark_z, cx, cy

    
def hand_detect(frame, previous_fingers:dict):
    """Draw hand_detection and returns if has a chance to release or play a note"""
    key_info = {}
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = True
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                rgb_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for fingertip_number in LANDMARK_FINGERTIPS_NUMBERS:
                fingertip_landmark = hand_landmarks.landmark[fingertip_number]
                cz, cx, cy = get_fingertip_landmarks(fingertip_landmark)
                cv2.circle(rgb_image, (cx, cy), 10, RGB_BLUE_COLOR, cv2.FILLED)

                # the location is a Decimal with something like 12 numbers after 0
                # mediapipe normalize the number but is really volatil
                normalize_num = 2
                old_num = previous_fingers.get(fingertip_number) + normalize_num

                pressed = True if old_num < cy else False

                key_info[fingertip_number] = {
                    "pressed": pressed,
                    "cz": cz,
                    "cx": cx,
                    "cy": cy
                }

    return rgb_image, key_info
