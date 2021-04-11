import cv2
import numpy as np
from playsound import playsound


class ComputerVisionCapture:
    has_finger_on_the_screen = False
    buffer = None

    def rescale_frame(self, frame, scale=0.5):
        """
            Resize the frame is good to have an improvement
            of the capability to read the image
        """
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        dimensions = int(frame_width*scale), int(frame_height*scale)
        
        return cv2.resize(frame, dimensions)
    
    def clean_image_noises(self, img):
        # converts the image to gray (because it's easier to find elements)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Removes the "noise" of the image (removes extra elements like bad lighting)
        # TODO: check the right number for the blur
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), cv2.BORDER_DEFAULT)
        return blurred_image

    def find_countours(self, img):
        # cv2.RETR_EXTERNAL
        contours, hierarchies = cv2.findContours(img.copy(), cv2.RETR_LIST,
                               cv2.CHAIN_APPROX_NONE)
        return contours, hierarchies 

    def read_camera(self):
        cv2_webcam_number = 0
        camera = cv2.VideoCapture(cv2_webcam_number)
        while True:
            ret, frame = camera.read()
            rescaled_image = self.rescale_frame(frame)
            cleaned_image = self.clean_image_noises(rescaled_image)
            # threshold makes above 225 color will be black, under 125 will be white
            # test thresh vs canny
            thresh = cv2.threshold(cleaned_image, 125, 255, cv2.THRESH_BINARY_INV)[1]
            
            # cv2.imshow('blurred_img', self.clean_image_noises(frame))
            cv2.imshow("Blurred image", cleaned_image)
            cv2.imshow("Thresholded image", thresh)
            contours, hierarchies = self.find_countours(thresh)

            
            if cv2.waitKey(1) == 27:
                break

        camera.release()
        cv2.destroyAllWindows()
