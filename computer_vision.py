import cv2
import numpy as np
from playsound import playsound


class ComputerVisionCapture:
    has_finger_on_the_screen = False

    def __init__(self, ratio=2.0):
        self.ratio = ratio

    def draw_contours(contours, text=None, img_to_draw, contour_color=(0,0,255), color_thickness=1):
        color_thickness = 1
        cv2.drawContours(img_to_draw, contours, -1, contour_color, color_thickness)
        if text:
            pass
            # cv2.putText(frame, shape, (cX, cY), 0.5, (255, 255, 255), 2)

    def is_rectangle(self, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        return True if len(approx) == 4 else False

    def rescale_frame(self, frame, scale=0.5):
        """
            Resize the frame is good to have an improvement
            of the capability to read the image
        """
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        dimensions = int(frame_width*scale), int(frame_height*scale)

        return cv2.resize(frame, dimensions)

    def clean_image_noises(self, img):
        # converts the image to gray (because it's easier to find elements)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Removes the "noise" of the image (removes extra elements like bad lighting)
        # TODO: check the right number for the blur
        blurred_image = cv2.GaussianBlur(
            gray_image, (5, 5), cv2.BORDER_DEFAULT)
        return blurred_image

    def find_countours(self, img):
        # cv2.RETR_EXTERNAL
        # TODO: test cv2.RETR_TREE instead RETR_LIST
        # TODO: test cv.CHAIN_APPROX_SIMPLE instead CHAIN_APPROX_NONE
        all_contours_found = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        # contours_with_good_area_size = []
        # remove small and large contours (small countor area certainly won't be a key)
        # for contour in all_contours_found:
        #     area = cv2.contourArea(contour, True)
        #     if 100 < area < 2000:
        #         contours_with_good_area_size.append(contour)

        return all_contours_found

    def read_camera(self):
        cv2_webcam_number = 0
        camera = cv2.VideoCapture(cv2_webcam_number)
        while True:
            _, frame = camera.read()
            rescaled_image = self.rescale_frame(frame)
            cleaned_image = self.clean_image_noises(rescaled_image)

            # threshold makes above 225 color will be black, under 125 will be white
            # test thresh vs canny
            thresh = cv2.threshold(cleaned_image, 125, 255, cv2.THRESH_BINARY_INV)[1]
            
            # https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
            contours = self.find_countours(thresh)

            self.draw_contours(cleaned_image, contours)
            self.draw_contours(rescaled_image, contours)
                

            cv2.imshow("Blurred image", cleaned_image)
            cv2.imshow("Thresholded image", thresh)
            cv2.imshow("Webcam normal rescaled image", rescaled_image)

            # press ESC to exit
            if cv2.waitKey(1) == 27:
                break

        camera.release()
        cv2.destroyAllWindows()
