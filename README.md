# Python Audio-Processing Web-Anwendung

## Dieses Projekt wird nicht mehr weiterentwickelt. Falls du Interesse daran hast, die Entwicklung weiterzuführen, kannst du einfach einen fork dieses Repository erstellen.

Eine interaktive Web-Anwendung zur Demonstration von digitaler Signalverarbeitung, gebaut mit Python, Flask und NumPy. Das Projekt ermöglicht die Synthese von Klängen, die Codierung von Text in ein robustes Audiosignal und die anschließende Decodierung zurück in Text mittels Mikrofonaufnahme und FFT-Analyse.

![Live-Demo](https://img.shields.io/badge/Live--Demo-Offline-lightgrey)
![Python Version](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Features

-   **Audio-Synthesizer:** Erzeuge einzelne Töne, Tonleitern oder Akkorde (Arpeggios) mit benutzerdefinierten Frequenzen.
-   **Visualisierung:** Lasse dir die Wellenform (Amplitude vs. Zeit) und das Frequenzspektrum (Amplitude vs. Frequenz via DFT/FFT) der erzeugten Klänge als Graphen anzeigen.
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

## 🚀 Live-Demo (Offline)

Die Live-Demo ist nicht mehr online. 

## Getting Started

Folge diesen Schritten, um das Projekt lokal auszuführen.

### Voraussetzungen

Stelle sicher, dass die folgenden Werkzeuge auf deinem System installiert sind:

-   Python 3.8+
-   pip (Python package installer)
-   **FFmpeg:** Eine kritische Abhängigkeit für die `pydub`-Bibliothek.
    -   **Windows:** [Download von der offiziellen Seite](https://ffmpeg.org/download.html) und füge das `bin`-Verzeichnis zu deiner PATH-Umgebungsvariable hinzu.
    -   **macOS:** `brew install ffmpeg`
    -   **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install ffmpeg`

### Installation
1. **Bei Bedarf Voraussetzungen installieren**
```sh
sudo apt update
sudo apt install python3-venv python3-pip
```

2.  **Repository klonen**
    ```sh
    git clone https://github.com/leon-weiss/Python-Audio-Processor.git
    cd Python-Audio-Processor
    ```

3.  **Virtuelle Umgebung erstellen und aktivieren**
    ```sh
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

4.  **Abhängigkeiten installieren**
    ```sh
    pip install -r requirements.txt
    ```

### Ausführung

Starte die Flask-Anwendung:
```sh
python app.py
```
Öffne deinen Browser und navigiere zu `http://localhost:5000`.

## Benutzung

Die Anwendung ist in drei Hauptbereiche unterteilt, die über die Navigation erreicht werden können:

1.  **Synthesizer:** Gib Frequenzen ein, um Töne, Tonleitern oder Akkorde zu erzeugen.
2.  **Text zu Musik:** Gib einen Text ein, um die entsprechende Audio-Datei für die Übertragung zu generieren.
3.  **Musik zu Text:** Führe den dreistufigen Decodierungs-Prozess durch:
    1.  **Kalibrierung:** Sorge für Stille und klicke auf "Kalibrierung starten", um das Rauschprofil deiner Umgebung zu erstellen.
    2.  **Aufnahme:** Spiele die zuvor generierte Audio-Datei ab und klicke auf "Aufnahme starten", um sie aufzunehmen. Stoppe die Aufnahme nach Ende des Signals.
    3.  **Decodierung:** Klicke auf "Decodieren", um das Signal zu analysieren und den Text zu rekonstruieren.

## Technischer Stack

-   **Backend:** Python, Flask
-   **Audio-Verarbeitung:** NumPy, SciPy, pydub
-   **Frontend:** HTML, CSS, JavaScript (mit der Web Audio API für die Aufnahme)
-   **Visualisierung:** Matplotlib

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe die Datei LICENSE für weitere Details.

## Beitrag leisten (Contributing)

Beiträge sind herzlich willkommen! Egal ob es sich um einen Bug-Report, eine neue Feature-Idee oder eine Code-Verbesserung handelt – jede Hilfe ist wertvoll.

### Wie du helfen kannst

-   **Fehler melden:** Du hast einen Fehler gefunden? Erstelle bitte ein [neues Issue](https://github.com/leon-weiss/Python-Audio-Processor/issues) und beschreibe den Fehler so genau wie möglich. Gib am besten auch an, welches Betriebssystem und welchen Browser du nutzt.
-   **Neue Ideen vorschlagen:** Wenn du eine Idee für eine neue Funktion oder eine Verbesserung hast, erstelle ebenfalls ein [Issue](https://github.com/leon-weiss/Python-Audio-Processor/issues). Beschreibe deine Idee und warum sie für das Projekt nützlich wäre.
-   **Code beisteuern:** Wenn du selbst Code beisteuern möchtest, folge bitte diesen Schritten:
    1.  Erstelle einen "Fork" dieses Repositories.
    2.  Erstelle einen neuen Branch für deine Änderungen (`git checkout -b feature/MeineTolleIdee`).
    3.  Nimm deine Änderungen vor und committe sie mit einer aussagekräftigen Nachricht.
    4.  Pushe deine Änderungen in deinen Fork.
    5.  Erstelle einen "Pull Request" zurück in dieses Repository und erkläre kurz, was deine Änderungen bewirken.


## Danksagung / Acknowledgements

Ein besonderer Dank gilt Prof. Panitz und dem Modul "Programmierparadigmen" an der Hochschule RheinMain. Eine Aufgabe in diesem Kurs zur Audioverarbeitung in Python lieferte die ursprüngliche Inspiration und den Anstoß für dieses Projekt.

Dieses Projekt baut zudem auf der Arbeit der Open-Source-Community auf. Ein Dank geht an die Entwickler und Mitwirkenden der folgenden Bibliotheken:

- [Flask](https://flask.palletsprojects.com/en/stable/)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [pydub](https://github.com/jiaaro/pydub)

---
