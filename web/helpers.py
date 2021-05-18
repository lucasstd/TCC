import cv2
import numpy as np

def get_cleaned_image_from_noises(image):
    # Removes the "noise" of the image (removes extra elements like bad lighting)
    blurred_image = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
    return blurred_image


def threshold_image(image):
    blurred_image=get_cleaned_image_from_noises(image)
    # converts the image to gray (because it's easier to find elements)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    # threshold makes above 225 color will be black, under 125 will be white
    thresh = cv2.threshold(gray_image, 150, 200, cv2.THRESH_BINARY_INV)[1]
    return thresh


def rescale_image(image, scale=0.5):
    """
        Resize the frame is good to have an improvement
        of the capability to read the image
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    dimensions = int(image_width*scale), int(image_height*scale)

    return cv2.resize(image, dimensions)
