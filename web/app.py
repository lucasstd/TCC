import json
import cv2

from flask import Flask, render_template, Response

from computer_vision import computer_vision


songs = ""
with open("songs.json", "r") as songs_file:
    songs = json.load(songs_file)


app = Flask(__name__)


# https://stackoverflow.com/questions/57762334/image-streaming-with-opencv-and-flask-why-imencode-is-needed
def get_contours_in_image():
    camera = cv2.VideoCapture(0)
    fingertips_indexes = [10, 10, 10, 10, 10]
    while True:
        processed_image, fingertips_indexes = computer_vision.process_image(
            fingertips_indexes, camera
        )

        _, buffer = cv2.imencode('.jpg', processed_image)
        buffered_image = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n%s\r\n" % buffered_image
        )


@app.route('/get-contours')
def get_contours_image():
    return Response(
        get_contours_in_image(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/')
def index():
    return render_template("index.html", all_songs=songs)


@app.route("/song/<song_id>")
def song_tutorial(song_id):
    return render_template('song.html', music_link="https://www.youtube.com/watch?v=7eQBm-j8Ev0")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
