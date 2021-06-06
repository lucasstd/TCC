import cv2
import config


def get_cleaned_image_from_noises(image):
    # Removes the "noises" in image (removes extra elements like bad lighting)
    blurred_image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    return blurred_image


def threshold_image(image):
    blurred_image = get_cleaned_image_from_noises(image)
    # converts the image to gray (because it's easier to find elements)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_image, 150, 200, cv2.THRESH_BINARY_INV)[1]
    # return cv2.Canny(gray_image, 10, 200, 3)
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
