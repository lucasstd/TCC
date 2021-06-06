import cv2
import numpy as np
import config
import hand_tracker

from play_sounds import play_note_by_key_place
import helpers


class ComputerVisionCapture:

    def __init__(self, ratio=2.0):
        self.ratio = ratio

    def is_a_keyboard_key(self, contour):
        contour_area = cv2.contourArea(contour)
        if 10000 > contour_area > 400:
            are_shapes_close = True
            contour_length = cv2.arcLength(contour, are_shapes_close)
            approx_curve = cv2.approxPolyDP(
                contour, 0.02 * contour_length, are_shapes_close)
            if 12 > len(approx_curve) > 3:
                return True
        return False

    # def find_fingers(self, image, min_color_bound=np.array([5,55,60], dtype=np.uint8),
    #                 max_color_bound=np.array([31, 255, 255], dtype=np.uint8),
    #                 kernel_open=np.ones((5, 5)),
    #                 kernel_close = np.ones((20, 20))):
    #     image_in_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     mask = cv2.inRange(image_in_hsv, min_color_bound, max_color_bound)
        # filters (It is useful in removing noise outside the main structure)
        # mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        # filters (It is useful in removing noise inside the main structure)
        # mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)

    # def draw_notes_name(self, image, contours):
    #     for contour in contours:
    #         M = cv2.moments(contour)
    #         if M["m00"] != 0:
    #             cx = int(M["m10"] / M["m00"] * self.ratio)
    #             cy = int(M["m01"] / M["m00"] * self.ratio)
    #             cv2.putText(image, f"nota: {contour}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
        # cv2.putText(frame, shape, (cX, cY), 0.5, (255, 255, 255), 2)
        # self.find_fingers(rescaled_image)

        # contours_hsv = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.imshow("Blurr", frame)

        # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def handle_empty_camera_frame(self) -> np.ndarray:
        """Returns a black image with can't read the camera drawn in blue"""
        empty_image = np.zeros(
            (config.IMG_SHAPE_Y, config.IMG_SHAPE_X, 3),
            np.uint8
        )
        cv2.putText(
            empty_image,
            "Can't read the camera",
            (130, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.THIN_LINE_SIZE,
            config.RGB_BLUE_COLOR,
            config.MEDIUM_LINE_SIZE
        )
        return empty_image

    
    def find_countours(self, img):
        all_contours_found, hierarchy = cv2.findContours(
            img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        return filter(self.is_a_keyboard_key, all_contours_found)
        notes = []
        for contour in all_contours_found:
            if self.is_a_keyboard_key(contour):
                notes.append(contour)
        return notes

    def draw_contours(self, img_to_draw):
        thresholded_image = helpers.threshold_image(img_to_draw)
        contours = self.find_countours(thresholded_image)
        if contours:
            try:
                # self.draw_notes_name(img_to_draw, contours)
                cv2.drawContours(
                    img_to_draw, contours, -1, config.RGB_RED_COLOR, 2
                )
            except Exception:
                pass  # contours null, but that's not an error
        return img_to_draw, thresholded_image

    def process_image(self, previous_indexes, frame=cv2.VideoCapture(0)):
        success, frame = frame.read()
        if not success:  # empty camera frame
            camera_error_img = self.handle_empty_camera_frame()
            return camera_error_img, previous_indexes

        rescaled_frame = helpers.rescale_image(frame)
        img_contours, threshold_img = self.draw_contours(rescaled_frame.copy())

        hand_detected, position_fingertip_played = hand_tracker.hand_detect(
            rescaled_frame, previous_indexes)

        stacked_img = np.hstack((img_contours, hand_detected))

        # play_note_by_key_place(1)
        return stacked_img, position_fingertip_played


computer_vision = ComputerVisionCapture()
