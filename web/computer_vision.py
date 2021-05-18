import cv2
import numpy as np
# import configparser

# from play_sounds import play_note_by_key_place
import helpers


class ComputerVisionCapture:

    def __init__(self, ratio=2.0):
        self.ratio = ratio

    def is_a_keyboard_key(self, contour):
        contour_area = cv2.contourArea(contour)
        if 10000 > contour_area > 200:
            are_shapes_close = True
            contour_length = cv2.arcLength(contour, are_shapes_close)
            approx_curve = cv2.approxPolyDP(contour, 0.02 * contour_length, are_shapes_close)
            if 9 > len(approx_curve) > 4:
                return True
        return False

    def find_fingers(self, image, min_color_bound=np.array([5,55,60], dtype=np.uint8),
                    max_color_bound=np.array([31, 255, 255], dtype=np.uint8),
                    kernel_open=np.ones((5, 5)),
                    kernel_close = np.ones((20, 20))):
        image_in_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image_in_hsv, min_color_bound, max_color_bound)
        # filters (It is useful in removing noise outside the main structure)
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        # filters (It is useful in removing noise inside the main structure)
        maskClose = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)
        # img_contours = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]


    def find_countours(self, img):
        """check https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html"""
        # TODO: test cv2.RETR_TREE instead RETR_LIST
        # TODO: test cv.CHAIN_APPROX_SIMPLE instead CHAIN_APPROX_NONE
        all_contours_found = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

        notes = []
        for contour in all_contours_found:
            if self.is_a_keyboard_key(contour):
                notes.append(contour)
        return notes
        # return filter(self.find_keys, all_contours_found)

    def draw_contours(self, img_to_draw):
        thresholded_image = helpers.threshold_image(img_to_draw)
        contours = self.find_countours(thresholded_image)
        if contours:
            try:
                self.draw_notes_name(img_to_draw, contours)
                cv2.drawContours(img_to_draw, contours, -1, (0,0,255), 2)
            except Exception: # contours null, but that's not an error
                pass
        return img_to_draw, thresholded_image

    def draw_notes_name(self, image, contours):
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"] * self.ratio)
                cy = int(M["m01"] / M["m00"] * self.ratio)
                cv2.putText(image, f"nota: {contour}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    # def read_rescaled_image(self, camera=None):
    #     if not camera:
    #         cv2_webcam_number = 0
    #         camera = cv2.VideoCapture(cv2_webcam_number)
    #     _, frame = camera.read()
    #     rescaled_image = self.rescale_frame(frame)
    #     return rescaled_image
    
    
    
        # cv2.putText(frame, shape, (cX, cY), 0.5, (255, 255, 255), 2)

        # cv2.imshow("Blurred image", blurred_image)
        # cv2.imshow("rescaled image", rescaled_image)
        # cv2.imshow("Thresholded image", thresh)
        # cv2.imshow("Gray image", gray_image)

        # self.find_fingers(rescaled_image)

        # contours_hsv = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.imshow("Blurr", frame)

        # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)



    def process_image(self, frame=cv2.VideoCapture(0)):
        _, frame = frame.read()
        rescaled_frame = helpers.rescale_image(frame)
        image_with_contours, thresholded_image = self.draw_contours(rescaled_frame)

        # img_stacked = np.hstack((image_with_contours, thresholded_image))
        return thresholded_image
        # play_note_by_key_place(1)
