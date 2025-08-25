# Student Question Simulator

This Gradio app simulates four enthusiastic students asking questions in response to a live recording of a lecture. The app creates an interactive educational experience where you can deliver a lecture and see real-time simulated student engagement.

## How It Works

- **Live Audio Recording**: Records lecture audio through your microphone in real-time
- **Automatic Sentence Detection**: Intelligently detects the end of each sentence without manual intervention
- **AI-Powered Student Questions**: After each sentence, prompts OpenAI's API with: "You are a student in an Introduction to Machine Learning course. Whenever the professor says something, you enthusiastically ask a question about it. Even if it would be inappropriate to interrupt with a question, you should still pose a question."
- **Multiple Student Perspectives**: Generates 4 different question samples from the same prompt to simulate diverse student thinking
- **Real-Time Streaming**: Each student's question is gradually revealed as it streams back from the API
- **Visual Interface**: Features four student avatars, each with their own text box showing their generated questions

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
