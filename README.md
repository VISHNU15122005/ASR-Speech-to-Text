# 🎙️ ASR-Speech-to-Text

### Speech-to-Text ASR Tool using Python and OpenAI Whisper

A web-based Automatic Speech Recognition (ASR) application that converts uploaded audio files into text using OpenAI Whisper and Streamlit.

---

## 📌 Project Description

This project is a locally running Speech-to-Text application. Users can upload audio files in WAV, MP3, M4A, or FLAC format. The application processes the audio using the Whisper ASR model and displays the generated transcription.

The transcript can also be downloaded as a `.txt` file.

---

## 🎯 Objectives

- Implement a working Automatic Speech Recognition tool.
- Convert speech into written text.
- Support common audio formats.
- Detect the spoken language.
- Display transcription and processing information.
- Allow users to download the transcript.

---

## ⚙️ How It Works

```text
Audio Input
     ↓
Audio Preprocessing
     ↓
Whisper ASR Model
     ↓
Speech Recognition
     ↓
Text Output
```

The uploaded audio is processed using FFmpeg and passed to the Whisper model. Whisper analyzes the audio and generates the corresponding text.

---

## ✨ Features

- Upload WAV, MP3, M4A and FLAC files
- Speech-to-text transcription using Whisper
- Automatic language detection
- Whisper model selection
- Audio playback
- Displays transcription and processing information
- Download transcript as `.txt`
- Error handling for invalid or missing files

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenAI Whisper | Automatic Speech Recognition |
| PyTorch | Deep Learning Framework |
| Streamlit | Web Interface |
| FFmpeg | Audio Processing |
| Git & GitHub | Version Control |
| VS Code | Development Environment |

---

## 📁 Project Structure

```text
ASR-Speech-to-Text/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
│
├── assets/
│   └── screenshots/
│
└── sample/
    └── README.md
```

---

## 💻 System Requirements

- Windows 10/11 (64-bit)
- Python 3.10 or 3.11
- FFmpeg
- 8 GB RAM recommended
- Internet connection for installation and initial model download

---

## 🚀 Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/VISHNU15122005/ASR-Speech-to-Text.git
cd ASR-Speech-to-Text
```

### 2. Create Virtual Environment

```powershell
python -m venv venv
```

Activate the environment:

```powershell
venv\Scripts\Activate.ps1
```

### 3. Upgrade pip

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Install FFmpeg

```powershell
winget install Gyan.FFmpeg
```

Verify the installation:

```powershell
ffmpeg -version
```

---

## ▶️ Run the Application

Start the Streamlit application:

```powershell
python -m streamlit run app.py
```

Open the URL shown in the terminal:

```text
http://localhost:8501
```

Upload an audio file and click **Transcribe Audio** to generate the transcript.

---

## 🧪 Testing

| Test Case | Input | Expected Result |
|---|---|---|
| Clear Speech | Clear English audio | Accurate transcription |
| Background Noise | Noisy speech | Transcription with possible errors |
| Invalid File | Non-audio file | Error message |
| No File | No audio selected | Error message |
| Multiple Formats | WAV, MP3, M4A, FLAC | Successful transcription |

---

## ✅ Advantages

- Free and locally runnable
- No API key required
- Supports multiple audio formats
- Supports multiple languages
- Simple and easy-to-use interface
- Suitable for speech transcription

---

## ⚠️ Limitations

- CPU processing can be slow for long audio
- Accuracy may decrease with heavy background noise
- Does not identify individual speakers
- Current version processes uploaded audio rather than live speech

---

## 🔮 Future Enhancements

- Real-time microphone transcription
- Speaker identification
- SRT/VTT subtitle generation
- PDF/DOCX transcript export
- Batch audio transcription
- GPU acceleration
- Automatic transcript summarization

---

## 📌 GitHub Repository

**Repository:**  
https://github.com/VISHNU15122005/ASR-Speech-to-Text

### Initial Upload

```powershell
git init
git add .
git commit -m "Initial ASR tool implementation"
git branch -M main
git remote add origin https://github.com/VISHNU15122005/ASR-Speech-to-Text.git
git push -u origin main
```

### Update Project

```powershell
git add .
git commit -m "Update ASR tool"
git push
```

---

## 👨‍💻 Author

**Name:** Vishnu B.  
**Register Number:** RA2311003050072  
**Course:** B.Tech Computer Science and Engineering  
**College:** SRM Institute of Science and Technology  
**GitHub:** https://github.com/VISHNU15122005

---

## 📜 License

This project is released under the MIT License.

---

## 🙏 Acknowledgements

- OpenAI Whisper
- Streamlit
- PyTorch
- FFmpeg
