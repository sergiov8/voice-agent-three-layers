"""Offline tests for the shared helpers. No API key or network needed."""

import json
import math
import pathlib
import struct
import tempfile
import unittest
import wave

from common.annotations import word_annotations, write_jsonl
from common.pcm import CHUNK_BYTES, RATE, EndOfSpeechDetector, rms
from common.transcription_config import (TranscriptionConfigError,
                                         build_transcription_config,
                                         check_audio_duration)


def tone(amplitude: int, ms: int = 100) -> bytes:
  n = RATE * ms // 1000
  return struct.pack(f"<{n}h", *(int(amplitude * math.sin(2 * math.pi * 440 * i / RATE))
                                  for i in range(n)))


class TranscriptionConfigTest(unittest.TestCase):

  def test_default_is_empty_verbatim(self):
    self.assertEqual(build_transcription_config(), {"transcription_config": {}})

  def test_smart_mode(self):
    cfg = build_transcription_config(smart=True, language_codes=[])
    self.assertEqual(cfg["transcription_config"], {"language_codes": [], "mode": "smart"})

  def test_verbatim_with_metadata(self):
    cfg = build_transcription_config(diarization=True, word_timestamps=True)
    self.assertEqual(cfg["transcription_config"]["mode"], {
        "type": "verbatim",
        "diarization_mode": "speaker",
        "timestamp_granularities": ["word"],
    })

  def test_custom_vocabulary_alone_is_allowed(self):
    cfg = build_transcription_config(custom_vocabulary=["BigQuery"])
    self.assertEqual(cfg["transcription_config"]["custom_vocabulary"], ["BigQuery"])

  def test_rejected_combinations(self):
    bad = [
        dict(smart=True, diarization=True),
        dict(smart=True, word_timestamps=True),
        dict(custom_vocabulary=["x"], diarization=True),
        dict(custom_vocabulary=["x"], word_timestamps=True),
        dict(custom_vocabulary=[str(i) for i in range(1001)]),
    ]
    for kwargs in bad:
      with self.subTest(kwargs=list(kwargs)):
        with self.assertRaises(TranscriptionConfigError):
          build_transcription_config(**kwargs)

  def test_duration_limits(self):
    check_audio_duration(59 * 60)
    check_audio_duration(29 * 60, diarization=True)
    with self.assertRaises(TranscriptionConfigError):
      check_audio_duration(31 * 60, word_timestamps=True)
    with self.assertRaises(TranscriptionConfigError):
      check_audio_duration(61 * 60)


class AnnotationsTest(unittest.TestCase):

  RESPONSE = {
      "steps": [{
          "type": "model_output",
          "content": [{
              "type": "text",
              "text": "Hello world",
              "annotations": [
                  {"type": "word_info", "text": "Hello", "speaker": "spk_1",
                   "start_offset": "0.100s", "end_offset": "0.450s"},
                  {"type": "other"},
                  {"type": "word_info", "text": "world", "speaker": "spk_1",
                   "start_offset": "0.500s", "end_offset": "0.850s"},
              ],
          }],
      }],
  }

  def test_extracts_word_info_only(self):
    words = list(word_annotations(self.RESPONSE))
    self.assertEqual([w["text"] for w in words], ["Hello", "world"])
    self.assertEqual(words[0], {"speaker": "spk_1", "start": "0.100s",
                                "end": "0.450s", "text": "Hello"})

  def test_no_annotations(self):
    self.assertEqual(list(word_annotations({"steps": None})), [])

  def test_write_jsonl(self):
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d, "out.jsonl")
      self.assertEqual(write_jsonl(word_annotations(self.RESPONSE), path), 2)
      lines = path.read_text().splitlines()
      self.assertEqual(json.loads(lines[1])["text"], "world")


class PcmTest(unittest.TestCase):

  def test_chunk_size_is_100ms(self):
    self.assertEqual(CHUNK_BYTES, 3200)
    self.assertEqual(len(tone(1000)), CHUNK_BYTES)

  def test_rms(self):
    self.assertEqual(rms(b""), 0.0)
    self.assertEqual(rms(tone(0)), 0.0)
    self.assertAlmostEqual(rms(tone(1000)), 1000 / math.sqrt(2), delta=5)

  def test_detector_fires_once_after_trailing_silence(self):
    det = EndOfSpeechDetector(threshold=300, silence_ms=300)
    leading_silence = [det.feed(tone(0)) for _ in range(5)]
    speech = [det.feed(tone(2000)) for _ in range(5)]
    trailing = [det.feed(tone(0)) for _ in range(6)]
    self.assertFalse(any(leading_silence))
    self.assertFalse(any(speech))
    self.assertEqual(trailing, [False, False, True, False, False, False])

  def test_short_pause_does_not_end_turn(self):
    det = EndOfSpeechDetector(threshold=300, silence_ms=300)
    pattern = [2000, 0, 0, 2000, 0, 0, 2000]
    self.assertFalse(any(det.feed(tone(a)) for a in pattern))

  def test_read_pcm_chunks_rejects_wrong_format(self):
    import asyncio
    from common.pcm import read_pcm_chunks
    with tempfile.TemporaryDirectory() as d:
      path = str(pathlib.Path(d, "stereo.wav"))
      with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(tone(1000) * 2)

      async def consume():
        return [c async for c in read_pcm_chunks(path, realtime=False)]

      with self.assertRaises(ValueError):
        asyncio.run(consume())


if __name__ == "__main__":
  unittest.main()
