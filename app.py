import os
import math
import uuid

import numpy
import itertools
import cmath
import io
import base64
import threading
import time
from flask import Flask, request, render_template, flash, redirect, url_for, session
from pydub import AudioSegment
from scipy.io import wavfile
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Flask App initialisieren ---
app = Flask(__name__)
app.secret_key = 'dies-ist-ein-sehr-geheimer-schluessel'

if not os.path.exists('static'): os.makedirs('static')
if not os.path.exists('temp_profiles'): os.makedirs('temp_profiles')


# ==============================================================================
# AUDIO-SYNTHESIZER
# ==============================================================================
def writeWav(fn, ds):
    sample_rate = 44100
    if not ds:
        wavfile.write(fn, sample_rate, numpy.array([], dtype=numpy.int16))
        return
    max_abs_val = max(abs(s) for s in ds) if ds else 0
    target_max_amp = 32767.0
    scaling_factor = 1.0
    if max_abs_val > target_max_amp:
        scaling_factor = target_max_amp / max_abs_val
    processed_samples = [numpy.int16(round(s * scaling_factor)) for s in ds]
    numpy_array = numpy.array(processed_samples, dtype=numpy.int16)
    wavfile.write(fn, sample_rate, numpy_array)


def pluggedTime(t, wv):
    samples = []
    sample_rate = 44100
    initial_amplitude = 10000
    for x_n in range(t):
        current_amplitude = initial_amplitude / (2 ** (x_n // 5000))
        x_in_formula = wv * x_n / sample_rate
        s_val = sum((1 / i) * math.sin(2 * math.pi * x_in_formula * i) for i in range(1, 11))
        samples.append(current_amplitude * s_val)
    return samples


def getWavFromFile(fn):
    try:
        rate, data = wavfile.read(fn)
        return list(data)
    except Exception as e:
        print(f"Fehler beim Lesen der WAV-Datei {fn}: {e}")
        return []


def dft(xs):
    N = len(xs)
    if N == 0: return []
    return [sum(xs[n] * cmath.exp(-1j * 2 * cmath.pi * k * n / N) for n in range(N)) / N for k in range(N)]


def create_plot_as_base64(plot_function, *args):
    plt.figure(figsize=(10, 4))
    plot_function(*args)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64


def _draw_waveform_plot(wav_path):
    sample_rate, data = wavfile.read(wav_path)
    time_array = numpy.linspace(0., len(data) / sample_rate, len(data))
    plt.plot(time_array, data, label="Wellenform")
    plt.title("Wellenform-Analyse")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)


def _draw_dft_plot(wav_path):
    ws = getWavFromFile(wav_path)
    N = 4410
    if len(ws) < N: N = len(ws)
    if N == 0: return

    dft_input = [complex(s, 0) for s in ws[:N]]
    dft_result = dft(dft_input)

    frequencies = numpy.fft.fftfreq(len(dft_result), 1 / 44100)
    amplitudes = [abs(c) for c in dft_result]
    num_bins_to_plot = N // 2

    plt.plot(frequencies[:num_bins_to_plot], amplitudes[:num_bins_to_plot], color='black')
    plt.title("DFT - Frequenzanalyse")
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("Amplitude")
    plt.grid(True)


def delayed_delete(filepath, delay_seconds):
    def task():
        time.sleep(delay_seconds)
        try:
            os.remove(filepath)
            print(f"Datei {filepath} wurde nach {delay_seconds}s gelöscht.")
        except OSError as e:
            print(f"Fehler beim Löschen der Datei {filepath}: {e}")

    thread = threading.Thread(target=task)
    thread.start()


# ==============================================================================
# FLASK ROUTEN
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    gen_type = request.form.get('gen_type')
    frequencies_str = request.form.get('frequencies')

    try:
        normalized_str = frequencies_str.replace(';', ',').replace(' ', '')
        if not normalized_str:
            raise ValueError("Das Eingabefeld ist leer.")

        frequencies = [float(f.strip()) for f in normalized_str.split(',') if f.strip()]
        if not frequencies:
            raise ValueError("Keine gültigen Frequenzen gefunden.")

    except ValueError:
        flash('Eingabefehler! Bitte geben Sie nur Zahlen ein, getrennt durch Kommas.', 'error')
        return redirect(url_for('index'))

    base_filename = f"audio_{int(time.time() * 1000)}"
    wav_filename = f"{base_filename}.wav"
    wav_filepath = os.path.join('static', wav_filename)

    final_samples = []
    if gen_type == 'single':
        final_samples = pluggedTime(44100, frequencies[0])
    elif gen_type == 'scale':
        all_notes = [pluggedTime(44100 // 2, f) for f in frequencies]
        final_samples = list(itertools.chain.from_iterable(all_notes))
    elif gen_type == 'chord':
        duration = 2 * 44100
        notes_with_offset = [([0.0] * 2000 * i) + pluggedTime(duration, freq) for i, freq in enumerate(frequencies)]
        final_samples = [sum(s) for s in itertools.zip_longest(*notes_with_offset, fillvalue=0.0)]

    writeWav(wav_filepath, final_samples)
    delayed_delete(wav_filepath, delay_seconds=900)

    waveform_b64 = create_plot_as_base64(_draw_waveform_plot, wav_filepath)
    dft_b64 = create_plot_as_base64(_draw_dft_plot, wav_filepath)

    return render_template('results.html',
                           wav_file=wav_filename,
                           waveform_plot_b64=waveform_b64,
                           dft_plot_b64=dft_b64)


# ==============================================================================
# TEXT-ZU-MUSIK / MUSIK-ZU-TEXT
# ==============================================================================

SAMPLE_RATE = 44100
BIT_DURATION_S = 0.15  # Dauer eines Bits in Sekunden
BIT_PAUSE_S = 0.07    # Kurze Pause zwischen Bits
BIT_SAMPLES = int(BIT_DURATION_S * SAMPLE_RATE)
PAUSE_SAMPLES = int(BIT_PAUSE_S * SAMPLE_RATE)

# Frequenz-Paare
ZERO_FREQS = (770, 1209)  # Frequenzen für eine 0
ONE_FREQS = (941, 1477)   # Frequenzen für eine 1
ALL_FREQS = ZERO_FREQS + ONE_FREQS


def find_peak_frequency(samples, rate):
    if len(samples) == 0:
        return 0

    fft_result = numpy.fft.fft(samples)

    num_samples = len(samples)
    magnitudes = numpy.abs(fft_result[:num_samples // 2])

    peak_index = numpy.argmax(magnitudes)

    frequencies = numpy.fft.fftfreq(num_samples, 1 / rate)
    peak_freq = frequencies[peak_index]

    return abs(peak_freq)


def create_dtmf_tone(freq_pair, duration_samples):
    t = numpy.linspace(0, duration_samples / SAMPLE_RATE, duration_samples, endpoint=False)
    wave1 = numpy.sin(2 * numpy.pi * freq_pair[0] * t)
    wave2 = numpy.sin(2 * numpy.pi * freq_pair[1] * t)
    combined_wave = (wave1 + wave2) / 2
    return numpy.int16(combined_wave * 16383)

def trim_silence(audio_data, threshold_factor=0.08, window_ms=20):
    if len(audio_data) == 0: return audio_data
    max_amp = numpy.max(numpy.abs(audio_data))
    if max_amp == 0: return audio_data
    normalized_data = audio_data / max_amp
    threshold = threshold_factor
    window_size = int(SAMPLE_RATE * window_ms / 1000)
    start_index = 0
    for i in range(0, len(normalized_data) - window_size, window_size):
        rms = numpy.sqrt(numpy.mean(normalized_data[i:i+window_size]**2))
        if rms > threshold:
            start_index = i; break
    end_index = len(normalized_data)
    for i in range(len(normalized_data) - window_size, 0, -window_size):
        rms = numpy.sqrt(numpy.mean(normalized_data[i:i+window_size]**2))
        if rms > threshold:
            end_index = i + window_size; break
    return audio_data[start_index:end_index]


@app.route('/text_to_music', methods=['GET', 'POST'])
def text_to_music():
    if request.method == 'POST':
        user_text = request.form.get('user_text', '')
        if not user_text:
            flash("Bitte geben Sie einen Text ein.", "error")
            return redirect(url_for('text_to_music'))

        # 1. Text zu Binär-String (8-Bit ASCII)
        binary_string = ''.join(format(ord(char), '08b') for char in user_text)

        # 2. DTMF-Tonfolge für String
        melody_samples = numpy.array([], dtype=numpy.int16)
        for bit in binary_string:
            freq_pair = ZERO_FREQS if bit == '0' else ONE_FREQS
            note = create_dtmf_tone(freq_pair, BIT_SAMPLES)
            pause = numpy.zeros(PAUSE_SAMPLES, dtype=numpy.int16)
            melody_samples = numpy.concatenate([melody_samples, note, pause])

        # 3. WAV speichern
        base_filename = f"text_audio_{int(time.time() * 1000)}"
        wav_filename = f"{base_filename}.wav"
        wav_filepath = os.path.join('static', wav_filename)

        wavfile.write(wav_filepath, SAMPLE_RATE, melody_samples)
        delayed_delete(wav_filepath, delay_seconds=900)

        return render_template('text_to_music.html', wav_file=wav_filename, generated_text=user_text,
                               binary_string=binary_string)

    return render_template('text_to_music.html', wav_file=None)


@app.route('/music_to_text')
def music_to_text():
    return render_template('music_to_text.html')


def decode_bit_from_chunk(samples, rate, noise_profile):
    if len(samples) < 100: return None

    fft_result = numpy.fft.fft(samples)
    magnitudes = numpy.abs(fft_result[:len(samples) // 2])
    frequencies = numpy.fft.fftfreq(len(samples), 1 / rate)[:len(samples) // 2]

    freq_energies = {}
    for freq in ALL_FREQS:
        target_idx = numpy.argmin(numpy.abs(frequencies - freq))

        # Normalisierte Energie aus Rauschprofil lesen
        noise_level = noise_profile[target_idx]

        # Energie in Signal berechnen
        signal_energy = numpy.mean(magnitudes[max(0, target_idx - 2): target_idx + 3])

        # Score ist, wie stark das Signal über dem Rauschen liegt
        freq_energies[freq] = signal_energy / noise_level if noise_level > 0 else signal_energy

    score_0 = freq_energies[ZERO_FREQS[0]] + freq_energies[ZERO_FREQS[1]]
    score_1 = freq_energies[ONE_FREQS[0]] + freq_energies[ONE_FREQS[1]]

    if score_0 > score_1 * 1.5:  # Score für 0 muss 50% höher sein als für 1
        return '0'
    elif score_1 > score_0 * 1.5:  # Score für 1 muss 50% höher sein als für 0
        return '1'
    return None


@app.route('/calibrate_noise_profile', methods=['POST'])
def calibrate_noise_profile():
    if 'audio_data' not in request.files:
        return {"error": "Keine Kalibrierungsdaten gefunden."}, 400
    try:
        audio_file = request.files['audio_data']
        sound = AudioSegment.from_file(audio_file).set_frame_rate(SAMPLE_RATE).set_channels(1)
        data = numpy.array(sound.get_array_of_samples(), dtype=numpy.float32)
        fft_result = numpy.fft.fft(data)
        magnitudes = numpy.abs(fft_result[:len(data) // 2])
        noise_profile = magnitudes + 1e-6

        profile_id = str(uuid.uuid4())
        filepath = os.path.join('temp_profiles', f"{profile_id}.npy")

        numpy.save(filepath, noise_profile)

        session['profile_id'] = profile_id

        print(f"Rauschprofil mit ID {profile_id} gespeichert.")
        return {"status": "ok"}
    except Exception as e:
        return {"error": f"Kalibrierung fehlgeschlagen: {e}"}, 500


@app.route('/decode_audio', methods=['POST'])
def decode_audio():
    if 'profile_id' not in session:
        return {"decoded_text": "[FEHLER: Bitte zuerst die Kalibrierung durchführen.]"}

    profile_id = session['profile_id']
    profile_filepath = os.path.join('temp_profiles', f"{profile_id}.npy")

    if not os.path.exists(profile_filepath):
        return {"decoded_text": f"[FEHLER: Kalibrierungsprofil nicht gefunden. Bitte erneut kalibrieren.]"}

    noise_profile = numpy.load(profile_filepath)

    if 'audio_data' not in request.files:
        return {"error": "Keine Audiodaten gefunden."}, 400

    try:
        audio_file = request.files['audio_data']
        sound = AudioSegment.from_file(audio_file).set_frame_rate(SAMPLE_RATE).set_channels(1)
        data = numpy.array(sound.get_array_of_samples(), dtype=numpy.float32)
    except Exception as e:
        os.remove(profile_filepath)
        session.pop('profile_id', None)
        return {"error": f"Audio-Konvertierung fehlgeschlagen: {e}"}, 500

    trimmed_data = trim_silence(data)

    binary_result = ""
    IN_NOTE = False
    note_buffer = []
    energy_threshold = numpy.max(numpy.abs(trimmed_data)) * 0.1
    window_size = int(SAMPLE_RATE * 0.02)
    for i in range(0, len(trimmed_data) - window_size, window_size):
        window = trimmed_data[i: i + window_size]
        rms = numpy.sqrt(numpy.mean(window ** 2))
        if IN_NOTE:
            if rms < energy_threshold and len(note_buffer) > BIT_SAMPLES * 0.7:
                bit = decode_bit_from_chunk(numpy.array(note_buffer), SAMPLE_RATE, noise_profile)
                if bit is not None: binary_result += bit
                note_buffer = []
                IN_NOTE = False
            else:
                note_buffer.extend(window)
        else:
            if rms > energy_threshold:
                IN_NOTE = True
                note_buffer.extend(window)
    if IN_NOTE and len(note_buffer) > BIT_SAMPLES * 0.7:
        bit = decode_bit_from_chunk(numpy.array(note_buffer), SAMPLE_RATE, noise_profile)
        if bit is not None: binary_result += bit

    decoded_text = ""
    for i in range(len(binary_result) // 8):
        byte = binary_result[i * 8: (i + 1) * 8]
        try:
            decoded_text += chr(int(byte, 2))
        except ValueError:
            decoded_text += '?'

    if not decoded_text:
        return {
            "decoded_text": "[Kein Text erkannt. Bitte Kalibrierung wiederholen oder Aufnahmebedingungen verbessern.]"}

    try:
        os.remove(profile_filepath)
        session.pop('profile_id', None)
        print(f"Rauschprofil mit ID {profile_id} wurde gelöscht.")
    except OSError as e:
        print(f"Fehler beim Löschen des Profils {profile_id}: {e}")

    return {"decoded_text": decoded_text, "binary_string": binary_result}


if __name__ == '__main__':
    app.run(debug=True)