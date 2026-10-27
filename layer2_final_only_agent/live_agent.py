"""Layer 2: interim transcripts drive the UI, only finalized transcripts reach the agent.

Streams a WAV file to gemini-3.5-transcribe-live as if it were a microphone. Interim
hypotheses are printed as a live caption line. Each finalized segment is handed to a
separate reasoning model, which is the only component allowed to decide what happens next.

Example:
  python -m layer2_final_only_agent.live_agent question.wav
"""

import argparse
import asyncio
import sys

from google import genai
from google.genai import types

from common.pcm import MIME_TYPE, read_pcm_chunks

TRANSCRIBE_MODEL = "gemini-3.5-transcribe-live"
DEFAULT_AGENT_MODEL = "gemini-3.8-flash"
AGENT_INSTRUCTION = "You are a voice agent. Reply in one or two short spoken sentences."


async def route_transcripts(session, on_final, on_interim=None) -> None:
  """Sends interim hypotheses to the UI callback and finalized segments to the agent."""
  async for response in session.receive():
    content = response.server_content
    if not content:
      continue
    if content.interim_input_transcription and on_interim:
      on_interim(content.interim_input_transcription.text)
    if content.input_transcription:
      final_text = (content.input_transcription.text or "").strip()
      if final_text:
        await on_final(final_text)


async def send_audio(session, path: str) -> None:
  async for chunk in read_pcm_chunks(path):
    await session.send_realtime_input(audio=types.Blob(data=chunk, mime_type=MIME_TYPE))
  await session.send_realtime_input(audio_stream_end=True)


async def run(path: str, agent_model: str) -> None:
  client = genai.Client()

  async def agent_turn(text: str) -> None:
    print(f"\n[final]  {text}")
    reply = await client.aio.interactions.create(
        model=agent_model, input=text, system_instruction=AGENT_INSTRUCTION)
    print(f"[agent]  {reply.output_text}")

  def show_caption(text: str) -> None:
    print(f"\r[interim] {text}", end="", flush=True)

  config = types.LiveConnectConfig(
      response_modalities=["TEXT"],
      input_audio_transcription=types.AudioTranscriptionConfig(mode="SMART"),
  )
  async with client.aio.live.connect(model=TRANSCRIBE_MODEL, config=config) as session:
    receiver = asyncio.create_task(route_transcripts(session, agent_turn, show_caption))
    await send_audio(session, path)
    # Leave time for the last finalized segment and the agent reply, then stop listening.
    await asyncio.sleep(5)
    receiver.cancel()


def main() -> int:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("audio", help="16 kHz mono 16-bit WAV file")
  parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
  args = parser.parse_args()
  asyncio.run(run(args.audio, args.agent_model))
  return 0


if __name__ == "__main__":
  sys.exit(main())
