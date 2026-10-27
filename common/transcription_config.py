"""Builds transcription settings for gemini-3.5-transcribe and rejects combinations the API refuses.

Limits come from the Gemini API audio transcription guide:
https://ai.google.dev/gemini-api/docs/transcribe
"""

MAX_CUSTOM_VOCABULARY = 1000
RECOMMENDED_CUSTOM_VOCABULARY = 100
MAX_AUDIO_SECONDS = 60 * 60
MAX_AUDIO_SECONDS_WITH_METADATA = 30 * 60


class TranscriptionConfigError(ValueError):
  """Raised for a configuration the transcription API would reject."""


def build_transcription_config(
    *,
    smart: bool = False,
    diarization: bool = False,
    word_timestamps: bool = False,
    custom_vocabulary: list[str] | None = None,
    language_codes: list[str] | None = None,
) -> dict:
  """Returns a generation_config dict for client.interactions.create.

  Args:
    smart: Smart transcription. Removes filler words and resolves self-corrections.
    diarization: Label speakers. Batch only, up to 8 speakers, 3+ experimental.
    word_timestamps: Word-level start and end offsets. Batch only.
    custom_vocabulary: Terms to bias recognition toward. Up to 1,000.
    language_codes: BCP-47 codes. An empty list means automatic detection.

  Raises:
    TranscriptionConfigError: For combinations the API rejects.
  """
  wants_metadata = diarization or word_timestamps
  if custom_vocabulary and wants_metadata:
    raise TranscriptionConfigError(
        "custom_vocabulary cannot be combined with diarization or word timestamps")
  if smart and wants_metadata:
    raise TranscriptionConfigError(
        "smart mode cannot be combined with diarization or word timestamps")
  if custom_vocabulary and len(custom_vocabulary) > MAX_CUSTOM_VOCABULARY:
    raise TranscriptionConfigError(
        f"custom_vocabulary supports up to {MAX_CUSTOM_VOCABULARY} terms, "
        f"got {len(custom_vocabulary)}")

  config: dict = {}
  if language_codes is not None:
    config["language_codes"] = list(language_codes)
  if custom_vocabulary:
    config["custom_vocabulary"] = list(custom_vocabulary)
  if smart:
    config["mode"] = "smart"
  elif wants_metadata:
    mode: dict = {"type": "verbatim"}
    if diarization:
      mode["diarization_mode"] = "speaker"
    if word_timestamps:
      mode["timestamp_granularities"] = ["word"]
    config["mode"] = mode
  return {"transcription_config": config}


def check_audio_duration(seconds: float, *, diarization: bool = False,
                         word_timestamps: bool = False) -> None:
  """Raises TranscriptionConfigError if the audio is longer than a batch request allows."""
  limit = (MAX_AUDIO_SECONDS_WITH_METADATA if diarization or word_timestamps
           else MAX_AUDIO_SECONDS)
  if seconds > limit:
    raise TranscriptionConfigError(
        f"audio is {seconds / 60:.1f} min, the limit for this configuration is "
        f"{limit // 60} min")
