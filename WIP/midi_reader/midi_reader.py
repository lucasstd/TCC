import time
import pretty_midi
import cv2


def read_midi(filename):
    midi_data = pretty_midi.PrettyMIDI(filename)
    import pdb
    pdb.set_trace()
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            duration = note.end - start
            pitch = note.pitch
            velocity = note.velocity / 128.
            score.append([start, duration, pitch, velocity, instrument.name])
    return score

read_midi("LVB_Sonate.mid")
# libfmp.c1.visualize_piano_roll(score, figsize=(8, 3), velocity_alpha=True)