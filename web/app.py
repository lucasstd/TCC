from flask import Flask, render_template, Response
import cv2

from computer_vision import ComputerVisionCapture


app = Flask(__name__)
computer_vision = ComputerVisionCapture()


def get_contours_in_image():
    camera = cv2.VideoCapture(0)
    while True:
        processed_image = computer_vision.process_image(camera)

        # https://stackoverflow.com/questions/57762334/image-streaming-with-opencv-and-flask-why-imencode-is-needed
        _, buffer = cv2.imencode('.jpg', processed_image)
        buffered_image = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n%s\r\n" % buffered_image
        )


@app.route('/get-contours')
def get_contours_image():
    return Response(get_contours_in_image(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000, debug=True)
