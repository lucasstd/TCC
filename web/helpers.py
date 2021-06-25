import cv2
import config


def get_cleaned_image_from_noises(image):
    # Removes the "noises" in image (removes extra elements like bad lighting)
    blurred_image = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
    return blurred_image


def draw_contours(img, contours, color=config.RGB_RED_COLOR, thickness=2) -> None:
    try:
        cv2.drawContours(img, contours, -1, color, 2)
    except Exception:
        pass  # contours are null, but that's not an error

def threshold_image(image):
    # converts the image to gray (because it's easier to find elements)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_image, 150, 200, cv2.THRESH_BINARY_INV)[1]
    # thresh = cv2.threshold(gray_image, 100, 150, cv2.THRESH_BINARY_INV)[1]
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
