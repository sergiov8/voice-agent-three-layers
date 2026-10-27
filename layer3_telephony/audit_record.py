"""Layer 3: turn a call recording into a speaker-labeled, word-timed audit record.

Runs batch transcription with diarization and word timestamps, then writes one JSON line
per word. The output loads directly into BigQuery or any warehouse for dispute review.
Diarization and timestamps cap a request at 30 minutes of audio.

Example:
  python -m layer3_telephony.audit_record call.wav --out call_audit.jsonl
"""

import argparse
import collections
import sys

from google import genai

from common.annotations import word_annotations, write_jsonl
from common.pcm import wav_duration_seconds
from common.transcription_config import (TranscriptionConfigError,
                                         build_transcription_config,
                                         check_audio_duration)

MODEL = "gemini-3.5-transcribe"


def main() -> int:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("audio", help="Call recording")
  parser.add_argument("--out", default="audit.jsonl", help="JSON Lines output path")
  parser.add_argument("--call-id", default=None, help="Stored on every record")
  args = parser.parse_args()

  generation_config = build_transcription_config(diarization=True, word_timestamps=True)
  if args.audio.lower().endswith(".wav"):
    try:
      check_audio_duration(wav_duration_seconds(args.audio), diarization=True)
    except TranscriptionConfigError as e:
      print(f"Split the recording first: {e}", file=sys.stderr)
      return 2

  client = genai.Client()
  audio_file = client.files.upload(file=args.audio)
  interaction = client.interactions.create(
      model=MODEL,
      input=[{"type": "audio", "uri": audio_file.uri, "mime_type": audio_file.mime_type}],
      generation_config=generation_config,
  )

  call_id = args.call_id or args.audio
  records = [{"call_id": call_id, **w} for w in word_annotations(interaction)]
  written = write_jsonl(records, args.out)
  per_speaker = collections.Counter(r["speaker"] or "unknown" for r in records)
  print(f"Wrote {written} words to {args.out}")
  for speaker, count in per_speaker.most_common():
    print(f"  {speaker}: {count} words")
  return 0


if __name__ == "__main__":
  sys.exit(main())
