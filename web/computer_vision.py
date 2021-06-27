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

    # https://stackoverflow.com/questions/57762334/image-streaming-with-opencv-and-flask-why-imencode-is-needed
    def get_contours_in_image(self):
        """Yields an image type to be send with draws and contours"""
        camera = cv2.VideoCapture(0)
        fingers_info = {finger: 0 for finger in config.LANDMARK_FINGERTIPS_NUMBERS}
        while True:
            processed_image, fingers_info = computer_vision.process_image(
                fingers_info, camera
            )

            _, buffer = cv2.imencode('.jpg', processed_image)
            buffered_image = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n%s\r\n" % buffered_image
            )

    def sort_contours_left_to_right(self, contours, img):
        # helpers.draw_contours(contours, img)
        contour_info = []
        # TODO: Sort while insert item inside the list of objects
        for contour in contours:
            moment = cv2.moments(contour)
            # gets the center of the contour
            if moment["m00"] > 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])

                contour_info.append({
                    "contour": contour,
                    "contour_x": cx,
                    "contour_y": cy,
                })
        sorted_contours = sorted(contour_info, key=lambda x:x["contour_x"])

        notes_length = len(config.NOTES) - 1
        note_index = -1
        for index, contour in enumerate(sorted_contours):
            note_index = note_index + 1 if note_index < notes_length else -1 
            key_note = config.NOTES[note_index]
            cv2.putText(
                img,
                f"{key_note}",
                (contour["contour_x"], contour["contour_y"]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                config.RGB_BLUE_COLOR,
                2
            )
            sorted_contours[index].update({"key_note": key_note})
        return sorted_contours

    def find_contours(self, img_to_get_contours) -> list:
        thresholded_image = helpers.threshold_image(img_to_get_contours)
        all_contours_found, _ = cv2.findContours(
            thresholded_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        return list(filter(self.is_a_keyboard_key, all_contours_found))

    def display_error_frame(self, display_text="Can't read the camera"):
        """Returns a black image with can't read the camera drawn in blue"""
        error_reading_camera = np.zeros(
            (config.IMG_SHAPE_Y, config.IMG_SHAPE_X, 3),
            np.uint8
        )
        cv2.putText(
            error_reading_camera,
            display_text,
            (130, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.THIN_LINE_SIZE,
            config.RGB_BLUE_COLOR,
            config.MEDIUM_LINE_SIZE
        )
        return error_reading_camera

    def check_if_pressed_key(self, indexes, fingertips_info, contours) -> None:
        pressed_locations = []
        for finger_key, finger_values in fingertips_info.items():
            indexes[finger_key] = finger_values["cy"]
            if finger_values["pressed"]:
                pressed_locations.append((finger_values["cx"], finger_values["cy"]))
                # play_note_by_key_place(1)

    def process_image(self, previous_indexes:dict, frame=cv2.VideoCapture(0)) -> tuple:
        success, frame = frame.read()
        if not success:
            return self.display_error_frame()

        rescaled_frame = helpers.rescale_image(frame)

        blurred_image = helpers.get_cleaned_image_from_noises(rescaled_frame)
        
        contours = self.find_contours(blurred_image)

        sorted_contours = self.sort_contours_left_to_right(contours, rescaled_frame)

        only_contours = [contour_info['contour'] for contour_info in sorted_contours]
        cv2.drawContours(rescaled_frame, only_contours, -1, config.RGB_RED_COLOR, 2)

        hand_detected_image, fingertips_info = hand_tracker.hand_detect(
            rescaled_frame, previous_indexes)

        self.check_if_pressed_key(previous_indexes, fingertips_info, contours)

        aux = helpers.threshold_image(rescaled_frame)

        stacked_img = aux  # np.hstack((rescaled_frame, hand_detected_image))

        return stacked_img, previous_indexes


computer_vision = ComputerVisionCapture()
