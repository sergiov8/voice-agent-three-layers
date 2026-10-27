"""Layer 1: transcribe a recorded file and pick a mode the API will accept.

Smart output and verbatim-with-metadata are mutually exclusive, and custom vocabulary
cannot be combined with speaker labels or timestamps. The configuration is validated
locally so a bad combination fails here with a clear message, not as an API error.

Examples:
  python -m layer1_api_constraints.transcribe_file call.wav --smart
  python -m layer1_api_constraints.transcribe_file call.wav --diarization --word-timestamps
  python -m layer1_api_constraints.transcribe_file call.wav --vocab Kubernetes --vocab BigQuery
"""

import argparse
import sys

from google import genai

from common.annotations import word_annotations
from common.pcm import wav_duration_seconds
from common.transcription_config import (TranscriptionConfigError,
                                         build_transcription_config,
                                         check_audio_duration)

MODEL = "gemini-3.5-transcribe"


def main() -> int:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("audio", help="Path to an audio file (wav, mp3, flac, m4a, ...)")
  parser.add_argument("--smart", action="store_true", help="Clean, formatted output")
  parser.add_argument("--diarization", action="store_true", help="Label speakers")
  parser.add_argument("--word-timestamps", action="store_true", help="Per-word offsets")
  parser.add_argument("--vocab", action="append", help="Custom vocabulary term, repeatable")
  parser.add_argument("--language", action="append",
                      help="BCP-47 code such as en-US, repeatable. Omit for auto-detect")
  args = parser.parse_args()

  try:
    generation_config = build_transcription_config(
        smart=args.smart,
        diarization=args.diarization,
        word_timestamps=args.word_timestamps,
        custom_vocabulary=args.vocab,
        language_codes=args.language,
    )
    if args.audio.lower().endswith(".wav"):
      check_audio_duration(wav_duration_seconds(args.audio), diarization=args.diarization,
                           word_timestamps=args.word_timestamps)
  except TranscriptionConfigError as e:
    print(f"Rejected before calling the API: {e}", file=sys.stderr)
    return 2

  client = genai.Client()
  audio_file = client.files.upload(file=args.audio)
  interaction = client.interactions.create(
      model=MODEL,
      input=[{"type": "audio", "uri": audio_file.uri, "mime_type": audio_file.mime_type}],
      generation_config=generation_config,
  )

  print(interaction.output_text)
  words = list(word_annotations(interaction))
  if words:
    print(f"\n{len(words)} annotated words. First ten:")
    for w in words[:10]:
      speaker = f"[{w['speaker']}] " if w["speaker"] else ""
      print(f"  {speaker}{w['start']} -> {w['end']}  {w['text']}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
