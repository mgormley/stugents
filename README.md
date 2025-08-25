# Student Question Simulator

This demo records a short lecture through the microphone and simulates four
students asking questions after each sentence. The app automatically
transcribes the audio, detects sentence boundaries, and uses the OpenAI API to
stream back four different questions. Each one appears in real time beneath a
student avatar.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_key_here
```

3. Run the app:

```bash
python app.py
```

Record some audio and watch the four student boxes populate with questions.
