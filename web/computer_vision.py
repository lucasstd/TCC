import cv2
import numpy as np
# from play_sounds import play_note_by_key_place
# import configparser


class ComputerVisionCapture:

    def __init__(self, ratio=2.0):
        self.ratio = ratio

    def is_rectangle(self, contour):
        contour_length = cv2.arcLength(contour, True)
        approx_curve = cv2.approxPolyDP(contour, 0.04 * contour_length, True)
        contour_area = cv2.contourArea(approx_curve)
        return True if len(approx_curve) == 4 and 10000 > contour_area > 200 else False
        if len(approx_curve) == 4 and 10000 > contour_area > 200:
            return True
            # cnt = approx_curve.reshape(-1, 2)
            # max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            # if max_cos < 0.1:
            #     return True
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
        # print(img_contours)

    def rescale_frame(self, image, scale=0.5):
        """
            Resize the frame is good to have an improvement
            of the capability to read the image
        """
        image_height = image.shape[0]
        image_width = image.shape[1]
        dimensions = int(image_width*scale), int(image_height*scale)

        return cv2.resize(image, dimensions)

    def get_cleaned_image_from_noises(self):
        rescaled_image = self.read_rescaled_image()
        # Removes the "noise" of the image (removes extra elements like bad lighting)
        blurred_image = cv2.GaussianBlur(rescaled_image, (3, 3), cv2.BORDER_DEFAULT)
        return blurred_image

    def find_countours(self, img):
        """check https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html"""
        # TODO: test cv2.RETR_TREE instead RETR_LIST
        # TODO: test cv.CHAIN_APPROX_SIMPLE instead CHAIN_APPROX_NONE
        all_contours_found = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        notes = []

        for contour in all_contours_found:
            if self.is_rectangle(contour):
                notes.append(contour)
        return notes

    def draw_contours(self, img_to_draw):
        thresholded_image = self.get_thresholded_image()
        contours = self.find_countours(thresholded_image)
        self.draw_notes_name(img_to_draw, contours)

        cv2.drawContours(img_to_draw, contours, -1, (0,0,255), 2)
        return img_to_draw

    def draw_notes_name(self, image, contours):
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"] * self.ratio)
                cy = int(M["m01"] / M["m00"] * self.ratio)
                cv2.putText(image, f"nota: {contour}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    def read_rescaled_image(self, camera=None):
        if not camera:
            cv2_webcam_number = 0
            camera = cv2.VideoCapture(cv2_webcam_number)
        _, frame = camera.read()
        rescaled_image = self.rescale_frame(frame)
        return rescaled_image
    
    def get_thresholded_image(self):
        blurred_image = self.get_cleaned_image_from_noises()
        # converts the image to gray (because it's easier to find elements)
        gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

        # TODO: test thresh vs canny
        # threshold makes above 225 color will be black, under 125 will be white
        thresh = cv2.threshold(gray_image, 150, 200, cv2.THRESH_BINARY_INV)[1]

        return thresh
    
        # cv2.putText(frame, shape, (cX, cY), 0.5, (255, 255, 255), 2)

        # cv2.imshow("Blurred image", blurred_image)
        # cv2.imshow("rescaled image", rescaled_image)
        # cv2.imshow("Thresholded image", thresh)
        # cv2.imshow("Gray image", gray_image)

        # self.find_fingers(rescaled_image)

        # contours_hsv = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.imshow("Blurr", frame)

        # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


        # play_note_by_key_place(1)
