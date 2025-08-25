import asyncio
import io
import re
import wave
from typing import List, Tuple

import gradio as gr
import numpy as np
from openai import AsyncOpenAI

SYSTEM_PROMPT = (
    "You are a student in an Introduction to Machine Learning course. "
    "Whenever the professor says something, you enthusiastically ask a question about it. "
    "Even if it would be inappropriate to interrupt with a question, you should still pose a question."
)

client = AsyncOpenAI()

def numpy_to_wav(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    """Convert a numpy audio array into a WAV file stored in memory."""
    buffer = io.BytesIO()
    audio_int16 = (audio * 32767).astype(np.int16)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    buffer.seek(0)
    return buffer


async def stream_student(idx: int, sentence: str, queue: asyncio.Queue) -> None:
    """Stream tokens for a single student and push them onto a queue."""
    async with client.responses.stream(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sentence},
        ],
        temperature=1.0,
    ) as stream:
        async for event in stream:
            if event.type == "response.output_text.delta":
                await queue.put((idx, event.delta))
        await queue.put((idx, None))


async def ask_questions(sentence: str):
    """Yield updated texts for four students asking questions about the sentence."""
    queue: asyncio.Queue[Tuple[int, str | None]] = asyncio.Queue()
    tasks = [asyncio.create_task(stream_student(i, sentence, queue)) for i in range(4)]
    texts = ["" for _ in range(4)]
    finished = 0
    while finished < 4:
        idx, token = await queue.get()
        if token is None:
            finished += 1
        else:
            texts[idx] += token
            yield tuple(texts)
    await asyncio.gather(*tasks)


async def lecture(audio: Tuple[int, np.ndarray]):
    """Transcribe audio, split into sentences, and stream student questions."""
    sample_rate, data = audio
    buffer = numpy_to_wav(data, sample_rate)
    transcription = await client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=buffer,
    )
    sentences = re.findall(r"[^.!?]+[.!?]", transcription.text)
    for sentence in sentences:
        # clear boxes for new sentence
        yield tuple(["" for _ in range(4)])
        async for update in ask_questions(sentence.strip()):
            yield update

def main():    
    with gr.Blocks() as demo:
        gr.Markdown("# Student Question Simulator")
        audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Lecture Audio")
        with gr.Row():
            outputs: List[gr.Textbox] = []
            for i in range(4):
                with gr.Column():
                    gr.HTML("<div style='font-size:172px;text-align:center'>🧑‍🎓</div>")
                    outputs.append(gr.Textbox(label=f"Student {i+1}", lines=3))
        audio_input.change(lecture, inputs=audio_input, outputs=outputs)
        
    demo.launch()

if __name__ == "__main__":
    main()
