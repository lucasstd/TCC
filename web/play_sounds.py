# from _thread import start_new_thread
import simpleaudio as sa


def play_note_by_key_place(piano_key: int) -> None:
    notes = { 1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "A", 7: "B" }
    wave_obj = sa.WaveObject.from_wave_file(f"sounds/{notes.get(piano_key, 'C')}.wav")
    wave_obj.play()
