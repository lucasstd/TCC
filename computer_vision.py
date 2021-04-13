import cv2
import numpy as np
from playsound import playsound


class ComputerVisionCapture:
    has_finger_on_the_screen = False
    ratio = 2.0

    def detect_shape(self, c):
        discovered_shape = ""
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            discovered_shape = "square" if ar > 0.94 and ar < 1.06 else "rectangle"
        return discovered_shape

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
        blurred_image = cv2.GaussianBlur(
            gray_image, (5, 5), cv2.BORDER_DEFAULT)
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
            thresh = cv2.threshold(
                cleaned_image, 125, 255, cv2.THRESH_BINARY_INV)[1]

            # cv2.imshow('blurred_img', self.clean_image_noises(frame))
            cv2.imshow("Blurred image", cleaned_image)
            cv2.imshow("Thresholded image", thresh)
            contours, hierarchies = self.find_countours(thresh)

            newShapes = []

            new_cnts = [c for c in contours if 300 < cv2.contourArea(c) < 4000]
            new_cnts = [c for c in new_cnts if cv2.contourArea(c, True) > 0]
            for c in range(len(new_cnts)):
                # compute the center of the contour
                M = cv2.moments(new_cnts[c])
                # if going to divide by 0 then skip
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"] * self.ratio)
                    cY = int(M["m01"] / M["m00"] * self.ratio)

                shape = self.detect_shape(new_cnts[c])

                new_cnts[c] = new_cnts[c].astype("float")
                new_cnts[c] *= self.ratio
                new_cnts[c] = new_cnts[c].astype("int")
                cv2.drawContours(frame, [new_cnts[c]], -1, (0, 255, 0), 2)
                cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)

            # press ESC to exit
            if cv2.waitKey(1) == 27:
                break

        camera.release()
        cv2.destroyAllWindows()
