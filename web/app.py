from flask import Flask, render_template, Response
import cv2

from computer_vision import ComputerVisionCapture


app = Flask(__name__)
computer_vision = ComputerVisionCapture()


def get_contours_in_image():
    camera = cv2.VideoCapture(0)
    while True:
        rescaled_image = computer_vision.read_rescaled_image(camera)
        image_with_contours = computer_vision.draw_contours(rescaled_image)

        ret, buffer = cv2.imencode('.jpg', image_with_contours)
        buffered_image = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffered_image + b'\r\n'
        )


@app.route('/get-contours')
def get_contours_image():
    return Response(get_contours_in_image(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000, debug=True)
