# voice-agent-three-layers

Companion code for the article **3 layers that take a voice agent from demo to production**. ARTICLE_LINK_PLACEHOLDER

A transcription model with a great word error rate is not a voice agent you can ship. These samples show the three layers that sit between the two, using [Gemini 3.5 Transcribe](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-transcribe) for speech-to-text.

| Layer | What breaks in production | Sample |
| --- | --- | --- |
| 1. API constraints | The batch and live endpoints have different feature sets, and some settings cannot be combined | [`layer1_api_constraints/`](layer1_api_constraints/) |
| 2. Transcript logic | An agent that acts on interim transcripts acts on words nobody said | [`layer2_final_only_agent/`](layer2_final_only_agent/) |
| 3. Telephony | Dead air from silence timeouts, raw PCM bridges and no audit trail | [`layer3_telephony/`](layer3_telephony/) |

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your-key   # from https://aistudio.google.com/apikey
```

The live samples read 16 kHz mono 16-bit WAV files. Convert anything else with:

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 input.wav
```

Run every sample from the repository root with `python -m`.

## Layer 1: pick a mode the API will accept

```bash
python -m layer1_api_constraints.transcribe_file call.wav --smart
python -m layer1_api_constraints.transcribe_file call.wav --diarization --word-timestamps
python -m layer1_api_constraints.transcribe_file call.wav --smart --diarization   # rejected locally
```

[`common/transcription_config.py`](common/transcription_config.py) encodes the documented rules:

- Smart mode cannot be combined with speaker diarization or word timestamps.
- Custom vocabulary cannot be combined with speaker diarization or word timestamps.
- Custom vocabulary accepts up to 1,000 terms. Results are best under 100.
- Batch audio is limited to 1 hour, or 30 minutes with diarization or timestamps.

Copy that one file into your own project and bad configurations fail with a clear message before they reach the API.

## Layer 2: interim for the UI, final for the agent

```bash
python -m layer2_final_only_agent.live_agent question.wav
uvicorn layer2_final_only_agent.token_server:app --port 8080
```

[`live_agent.py`](layer2_final_only_agent/live_agent.py) streams audio to `gemini-3.5-transcribe-live`. Interim hypotheses are printed as a live caption. Only finalized segments are sent to the reasoning model.

[`token_server.py`](layer2_final_only_agent/token_server.py) mints single-use ephemeral tokens locked to the transcription model, so browser and mobile clients never hold your API key. Add authentication for your own users before deploying it.

## Layer 3: the phone call and the audit trail

```bash
python -m layer3_telephony.hybrid_vad_sender utterance.wav --threshold 400
python -m layer3_telephony.audit_record call.wav --out call_audit.jsonl
```

[`hybrid_vad_sender.py`](layer3_telephony/hybrid_vad_sender.py) sends audio in 100 ms chunks and fires `audio_stream_end` as soon as a local energy detector hears silence. The server's own VAD stays on as the fallback. The script prints how long the final transcript took after the end of speech.

[`audit_record.py`](layer3_telephony/audit_record.py) transcribes a recording with speaker labels and word timestamps and writes one JSON line per word, ready to load into BigQuery.

For real phone lines, a media framework such as [LiveKit](https://docs.livekit.io/agents/models/stt/gemini/) or [Pipecat](https://docs.pipecat.ai/api-reference/server/services/stt/google) handles SIP and RTP and hands you the PCM stream these samples expect.

## Tests

```bash
python -m unittest discover -s tests -t .
```

The tests cover the shared helpers and run offline with no API key.

## Notes

- **Gemini Enterprise Agent Platform.** The same model is available there as `gemini-3.5-transcribe-preview` and `gemini-3.5-transcribe-live-preview`, using `generate_content` for batch requests and a 15 minute file limit during preview. See the [Agent Platform model page](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-transcribe). These samples target the Gemini API.
- **Speech-to-speech alternative.** [Gemini 3.8 Live and 3.8 Live Extended Thinking](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-8-live-gemini-3-8-live-extended-thinking/) listen, reason and speak in one model. Choose that path for the lowest latency. Choose this pipeline when you need a text transcript you can inspect, audit and route to your own orchestrator. Layer 3 applies to both.
- **One utterance per run.** The hybrid VAD sample ends after the first finalized segment. Confirm that your session accepts more audio after `audio_stream_end` before reusing it for multi-turn calls.

## License

Apache 2.0. See [LICENSE](LICENSE). This is sample code, not an officially supported Google product.
