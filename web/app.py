import json

from flask import Flask, render_template, Response, abort

from computer_vision import computer_vision


songs = ""
with open("songs.json", "r") as songs_file:
    songs = json.load(songs_file)


app = Flask(__name__)


@app.route('/get-contours')
def get_contours_image():
    return Response(
        computer_vision.get_contours_in_image(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/')
def index():
    return render_template("index.html", all_songs=songs)


@app.route("/song/<song_id>")
def song_tutorial(song_id):
    # song = songs.get(song_id)
    for song_difficulty in songs:
        song_details = songs[song_difficulty].get(song_id)
        if song_details:
            return render_template('song.html', song=song_details)
    abort(404)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
