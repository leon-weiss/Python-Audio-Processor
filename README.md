# Python Audio-Processing Web-Anwendung

Eine interaktive Web-Anwendung zur Demonstration von digitaler Signalverarbeitung, gebaut mit Python, Flask und NumPy. Das Projekt ermöglicht die Synthese von Klängen, die Codierung von Text in ein robustes Audiosignal und die anschließende Decodierung zurück in Text mittels Mikrofonaufnahme und FFT-Analyse.

## Features

-   **Audio-Synthesizer:** Erzeugen Sie einzelne Töne, Tonleitern oder Akkorde (Arpeggios) mit benutzerdefinierten Frequenzen.
-   **Visualisierung:** Lassen Sie sich die Wellenform (Amplitude vs. Zeit) und das Frequenzspektrum (Amplitude vs. Frequenz via DFT/FFT) der erzeugten Klänge als Graphen anzeigen.
-   **Text-zu-Musik (Robuste Codierung):**
    -   Wandelt eine beliebige Texteingabe in eine Binärsequenz um.
    -   Codiert jedes Bit als eine robuste Dual-Ton-Frequenzkombination (inspiriert von DTMF).
    -   Gibt eine abspielbare `.wav`-Datei aus, die für die Übertragung über Lautsprecher und Mikrofon optimiert ist.
-   **Musik-zu-Text (Adaptive Decodierung):**
    -   Ein mehrstufiger Prozess zur Rückumwandlung des Audiosignals in Text.
    -   **Kalibrierung:** Ein initialer Schritt, um ein Rauschprofil der Aufnahmeumgebung (Lautsprecher, Raum, Mikrofon) zu erstellen und so die Genauigkeit drastisch zu erhöhen.
    -   **Aufnahme:** Nimmt das abgespielte Audiosignal über das Browser-Mikrofon auf.
    -   **Analyse:** Eine intelligente Decodierungs-Pipeline, die Folgendes beinhaltet:
        -   Automatisches Entfernen von Stille am Anfang und Ende.
        -   Energiebasierte Segmentierung zur dynamischen Erkennung einzelner Töne.
        -   Analyse jedes Tons mittels Fast-Fourier-Transformation (FFT).
        -   Decodierung der Bits basierend auf den normalisierten Frequenz-Scores (Signal vs. Rauschprofil).
        -   Rückumwandlung der erkannten Binärsequenz in Text.

## Getting Started

Folgen Sie diesen Schritten, um das Projekt lokal auszuführen.

### Voraussetzungen

Stellen Sie sicher, dass die folgenden Werkzeuge auf Ihrem System installiert sind:

-   Python 3.8+
-   pip (Python package installer)
-   **FFmpeg:** Eine kritische Abhängigkeit für die `pydub`-Bibliothek.
    -   **Windows:** [Download von der offiziellen Seite](https://ffmpeg.org/download.html) und fügen Sie das `bin`-Verzeichnis zu Ihrer PATH-Umgebungsvariable hinzu.
    -   **macOS:** `brew install ffmpeg`
    -   **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install ffmpeg`

### Installation

1.  **Repository klonen**
    ```sh
    git clone git@github.com:leon-weiss/Python-Audio-Processor.git
    cd Python-Audio-Processor
    ```

2.  **Virtuelle Umgebung erstellen und aktivieren**
    ```sh
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Abhängigkeiten installieren**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Temporäre Ordner erstellen**
    Erstellen Sie die folgenden leeren Ordner im Hauptverzeichnis, falls sie nicht existieren:
    -   `static`
    -   `temp_profiles`

### Ausführung

Starten Sie die Flask-Anwendung:
```sh
python app.py
```
Öffnen Sie Ihren Browser und navigieren Sie zu `http://127.0.0.1:5000`.

## Benutzung

Die Anwendung ist in drei Hauptbereiche unterteilt, die über die Navigation erreicht werden können:

1.  **Synthesizer:** Geben Sie Frequenzen ein, um Töne, Tonleitern oder Akkorde zu erzeugen.
2.  **Text zu Musik:** Geben Sie einen Text ein, um die entsprechende Audio-Datei für die Übertragung zu generieren.
3.  **Musik zu Text:** Führen Sie den dreistufigen Decodierungs-Prozess durch:
    1.  **Kalibrierung:** Sorgen Sie für Stille und klicken Sie auf "Kalibrierung starten", um das Rauschprofil Ihrer Umgebung zu erstellen.
    2.  **Aufnahme:** Spielen Sie die zuvor generierte Audio-Datei ab und klicken Sie auf "Signal-Aufnahme starten", um sie aufzunehmen. Stoppen Sie die Aufnahme nach Ende des Signals.
    3.  **Decodierung:** Klicken Sie auf "Decodieren", um das Signal zu analysieren und den Text zu rekonstruieren.

## Technischer Stack

-   **Backend:** Python, Flask
-   **Audio-Verarbeitung:** NumPy, SciPy, pydub
-   **Frontend:** HTML, CSS, JavaScript (mit der Web Audio API für die Aufnahme)
-   **Visualisierung:** Matplotlib

## Danksagung / Acknowledgements

Dieses Projekt wäre ohne die folgenden fantastischen Open-Source-Bibliotheken nicht möglich:

- [Flask](https://flask.palletsprojects.com/en/stable/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [pydub](https://github.com/jiaaro/pydub)

Vielen Dank an alle Entwickler und Mitwirkenden dieser Projekte!

---
