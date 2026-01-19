# modules/tts.py
import edge_tts
import os
import time

DEFAULT_VOICE = "en-US-AriaNeural"
DEFAULT_RATE  = "+0%"
DEFAULT_PITCH = "+0Hz"

TMP_DIR = "_tts_tmp"

async def synthesize_mp3_async(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    pitch: str = DEFAULT_PITCH,
) -> bytes:
    os.makedirs(TMP_DIR, exist_ok=True)

    tmp_path = os.path.join(
        TMP_DIR, f"edge_tts_{int(time.time()*1000)}.mp3"
    )

    communicator = edge_tts.Communicate(
        text=text, voice=voice, rate=rate, pitch=pitch
    )

    try:
        await communicator.save(tmp_path)

        with open(tmp_path, "rb") as f:
            return f.read()

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
