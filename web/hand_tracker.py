import cv2
import mediapipe as mp
from decimal import Decimal

from config import *

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
    max_num_hands=2
)


def draw_circle_in_fingertip(img_to_draw, fingertip_landmark):
    """convert from Decimal position to pixels and draw fingertips"""
    landmark_x = fingertip_landmark.x
    landmark_y = fingertip_landmark.y

    cx = int(IMG_SHAPE_X * landmark_x)
    cy = int(IMG_SHAPE_Y * landmark_y)

    cv2.circle(img_to_draw, (cx, cy), 10, RGB_BLUE_COLOR, cv2.FILLED)

    
def hand_detect(frame, previous_z_index=Decimal(10)):
    """Draw hand_detection and returns if has a chance to release or play a note"""
    fingertip_could_have_pressed_a_key = []
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                rgb_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for fingertip_number in LANDMARK_FINGERTIPS_NUMBERS:
                fingertip_landmark = hand_landmarks.landmark[fingertip_number]
                draw_circle_in_fingertip(rgb_image, fingertip_landmark)

                # if previous_z_index > fingertip_landmark.z:
                #     fingertip_could_have_pressed_a_key.append(True)
                # else:
                #     fingertip_could_have_pressed_a_key.append(False)

    return rgb_image, fingertip_could_have_pressed_a_key
