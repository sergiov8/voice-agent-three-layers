"""Layer 2: mint single-use ephemeral tokens so browser and mobile clients never hold your API key.

Each token works for one session, expires in 30 minutes and is locked to
gemini-3.5-transcribe-live with a text-only transcription config.

Run:
  uvicorn layer2_final_only_agent.token_server:app --port 8080
Then from the client:
  POST http://localhost:8080/token  ->  {"token": "...", "expires": "..."}
The client passes the token in place of an API key when it opens the Live API session.
Put authentication for your own users in front of this endpoint before deploying it.
"""

import datetime

from fastapi import FastAPI
from google import genai

TRANSCRIBE_MODEL = "gemini-3.5-transcribe-live"
TOKEN_LIFETIME = datetime.timedelta(minutes=30)

app = FastAPI()
client = genai.Client()


@app.post("/token")
def create_token() -> dict:
  expire_time = datetime.datetime.now(tz=datetime.timezone.utc) + TOKEN_LIFETIME
  token = client.auth_tokens.create(
      config={
          "uses": 1,
          "expire_time": expire_time,
          "live_connect_constraints": {
              "model": TRANSCRIBE_MODEL,
              "config": {
                  "response_modalities": ["TEXT"],
                  "input_audio_transcription": {"language_codes": []},
              },
          },
      })
  return {"token": token.name, "expires": expire_time.isoformat()}
