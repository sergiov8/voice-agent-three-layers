"""Extracts word-level speaker and timing annotations from a batch transcription result."""

import json
import pathlib
from collections.abc import Iterable, Iterator
from typing import Any


def _get(obj: Any, key: str) -> Any:
  """Reads a field from an SDK object or from a parsed REST response dict."""
  if isinstance(obj, dict):
    return obj.get(key)
  return getattr(obj, key, None)


def word_annotations(interaction: Any) -> Iterator[dict]:
  """Yields one dict per recognized word: speaker, start, end and text.

  Annotations are present only when diarization or word timestamps were requested.
  """
  for step in _get(interaction, "steps") or []:
    for content in _get(step, "content") or []:
      for ann in _get(content, "annotations") or []:
        if _get(ann, "type") != "word_info":
          continue
        yield {
            "speaker": _get(ann, "speaker"),
            "start": _get(ann, "start_offset"),
            "end": _get(ann, "end_offset"),
            "text": _get(ann, "text"),
        }


def write_jsonl(records: Iterable[dict], path: str | pathlib.Path) -> int:
  """Writes records as JSON Lines and returns how many were written."""
  count = 0
  with open(path, "w", encoding="utf-8") as f:
    for record in records:
      f.write(json.dumps(record, ensure_ascii=False) + "\n")
      count += 1
  return count
