# import ssl, certifi
# ssl._create_default_https_context = ssl._create_unverified_context

# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import whisper
# import os
# import sys
# import types
# import ffmpeg

# # ---- Patch audioop for Python 3.13 ----
# if "audioop" not in sys.modules:
#     sys.modules["audioop"] = types.ModuleType("audioop")
#     def fake_fn(*args, **kwargs): return None
#     for fn in ["add", "mul", "bias", "max", "avg", "reverse", "findfactor", "cross",
#                "getsample", "getsample_size", "lin2adpcm", "adpcm2lin"]:
#         setattr(sys.modules["audioop"], fn, fake_fn)

# from pydub import AudioSegment

# app = FastAPI()

# # ---- CORS ----
# origins = [
#     "http://localhost:5500",
#     "http://127.0.0.1:5500",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ---- Convert audio ----
# def convert_to_wav(input_path, output_path="converted.wav"):
#     ffmpeg.input(input_path).output(
#         output_path,
#         format="wav",
#         acodec="pcm_s16le",
#         ac=1,
#         ar="16000"
#     ).overwrite_output().run(quiet=True)

#     return output_path

# # ---- Load HuggingFace Gloss Model ----
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# model_path = "asl_t5_final"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
# def get_gloss(text):
#     inputs = tokenizer(text, return_tensors="pt")
#     output = model.generate(**inputs, max_length=128)
#     gloss = tokenizer.decode(output[0], skip_special_tokens=True)
#     return gloss

# print(get_gloss("I am going to school today."))




# import json

# with open("gloss_map.json", "r") as f:
#     gloss_map = json.load(f)

# import cv2

# def test_video(gloss):
#     path = gloss_map[gloss]
#     cap = cv2.VideoCapture(path)
#     ok, frame = cap.read()
#     cap.release()
#     print(gloss, "→", "OK" if ok and frame is not None else "BROKEN")

# test_video("I")
# test_video("GO")
# test_video("SCHOOL")

# import cv2
# import os
# import numpy as np

# def concat_videos_opencv(sentence,
#                                 output="asl_smooth.mp4",
#                                 fade_duration=0.25,
#                                 gap_duration=0.1,
#                                 height=480,
#                                 default_fps=25,
#                                 verbose=True):
#     """
#     ASL concatenator with:
#       - uniform frame size via padding
#       - smooth crossfade between signs
#       - optional black gap
#       - OpenCV-only
#     """

#     words = sentence.strip().upper().split()

#     # collect valid videos
#     video_paths = []
#     fps_candidates = []

#     for w in words:
#         if w not in gloss_map:
#             if verbose: print(f"[Missing] {w}")
#             continue

#         p = gloss_map[w]
#         if not os.path.exists(p):
#             if verbose: print(f"[File not found] {p}")
#             continue

#         video_paths.append((w, p))

#         cap = cv2.VideoCapture(p)
#         if cap.isOpened():
#             f = cap.get(cv2.CAP_PROP_FPS)
#             if f and f > 1:
#                 fps_candidates.append(f)
#         cap.release()

#     if not video_paths:
#         print("No valid videos.")
#         return

#     # use average fps
#     fps = int(sum(fps_candidates) / len(fps_candidates)) if fps_candidates else default_fps
#     if verbose: print("Chosen FPS:", fps)

#     # PASS 1 — determine output width
#     max_width = 0
#     for gloss, p in video_paths:
#         cap = cv2.VideoCapture(p)
#         ret, frame = cap.read()
#         cap.release()
#         if not ret: continue

#         h, w = frame.shape[:2]
#         new_w = int((height / h) * w)
#         max_width = max(max_width, new_w)

#     if verbose: print("Output width:", max_width)

#     # Prepare writer
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(output, fourcc, fps, (max_width, height))

#     def pad_frame(frame):
#         """Resize → pad frame to (max_width, height)."""
#         h0, w0 = frame.shape[:2]
#         frame = cv2.resize(frame, (int((height / h0) * w0), height))
#         h0, w0 = frame.shape[:2]

#         if w0 < max_width:
#             pad_left = (max_width - w0) // 2
#             pad_right = max_width - w0 - pad_left
#             frame = cv2.copyMakeBorder(frame, 0, 0, pad_left, pad_right,
#                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
#         else:
#             frame = frame[:, (w0 - max_width)//2:(w0 - max_width)//2 + max_width]

#         return frame

#     total_frames = 0
#     prev_last_frame = None
#     fade_frames = int(fade_duration * fps)
#     gap_frames = int(gap_duration * fps)

#     # PASS 2 — write all videos with smooth transitions
#     for idx, (gloss, p) in enumerate(video_paths):
#         cap = cv2.VideoCapture(p)
#         if verbose: print(f"[Writing] {gloss}")

#         frames = []
#         while True:
#             ret, frame = cap.read()
#             if not ret: break
#             frames.append(pad_frame(frame))
#         cap.release()

#         if len(frames) == 0: continue

#         # If NOT the first clip → fade transition from previous
#         if prev_last_frame is not None:
#             for i in range(fade_frames):
#                 alpha = 1 - (i / fade_frames)
#                 blended = (alpha * prev_last_frame + (1 - alpha) * frames[0]).astype(np.uint8)
#                 out.write(blended)
#                 total_frames += 1

#             # optional black gap
#             black = np.zeros((height, max_width, 3), dtype=np.uint8)
#             for _ in range(gap_frames):
#                 out.write(black)

#         # write full current clip
#         for fr in frames:
#             out.write(fr)
#             total_frames += 1

#         prev_last_frame = frames[-1]

#     out.release()

#     print(f"[Saved] {output} ({total_frames} frames written)")
# # concat_videos_opencv("I GO SCHOOL TOMORROW. HOW ARE YOU?", output="testing.mp4")









# # ---- Whisper ----
# model = whisper.load_model("base")
# def audio_to_text(audio_path: str) -> str:
#     wav_path = convert_to_wav(audio_path)
    
#     result = model.transcribe(wav_path)
#     return result["text"]

# # ---- API ----
# @app.post("/transcribe/")
# async def transcribe_audio(file: UploadFile = File(...)):
#     temp_path = f"temp_{file.filename}"
#     with open(temp_path, "wb") as f:
#         f.write(await file.read())

#     try:
#         text = audio_to_text(temp_path)
#         print(text)
#         print("check1")
#         gloss = get_gloss("who are you")

#         # gloss = get_gloss(text)
#         print("check3")
#         print(gloss)
        
#         concat_videos_opencv(gloss,output="testing.mp4")
#         # words=gloss.split()
#         # for i in words:
#         #     test_video(i)
#         # print("GLOSS:", gloss)
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)
#     finally:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#         if os.path.exists("converted.wav"):
#             os.remove("converted.wav")

#     return JSONResponse(content={"text": text, "gloss": gloss})




# # from fastapi.responses import FileResponse

# # @app.get("/video/{filename}")
# # def get_video(filename: str):
# #     path = filename
# #     if not os.path.exists(path):
# #         return JSONResponse(content={"error": "File not found"}, status_code=404)
# #     return FileResponse(path, media_type="video/mp4")




# # ---- Run Server ----
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)





# app.py  — improved version
import os
import sys
import uuid
import json
import traceback
import ssl
import types
import time
import re
# ======= environment & ssl helpers ========
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

# Prevent tokenizers parallelism warning after forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- Patch audioop for Python 3.13 if missing ----
if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")
    def _fake(*args, **kwargs): return None
    for fn in ["add", "mul", "bias", "max", "avg", "reverse", "findfactor", "cross",
               "getsample", "getsample_size", "lin2adpcm", "adpcm2lin"]:
        setattr(sys.modules["audioop"], fn, _fake)

# ======= FastAPI imports ========
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import ffmpeg
from pydub import AudioSegment

# ======= Machine learning libs ========
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ======= Utility imports ========
import cv2
import numpy as np
from pathlib import Path

# ======= App setup ========
app = FastAPI(title="Speech → Gloss → Sign Video")

# CORS — add your frontend origins
# origins = [
#     "http://localhost:5500",
#     "http://127.0.0.1:5500",
# ]
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:5501",
    "http://127.0.0.1:5501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======= Config / paths ========
MODEL_PATH = "asl_t5_final"   # change if your model folder is different
GLOSS_MAP_PATH = "gloss_map.json"  # expected file from earlier step
VIDEO_OUTPUT_DIR = Path("video")
TEMP_DIR = Path("temp_files")
VIDEO_OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ======= Load models once at startup ========
try:
    # Load whisper once
    whisper_model = whisper.load_model("base")
except Exception as e:
    print("Failed to load Whisper model:", e)
    raise

try:
    # Force slow tokenizer to avoid protobuf / fast-tokenizer conversion issues
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
    gloss_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True)
except Exception as e:
    print("Failed to load gloss model/tokenizer. Check MODEL_PATH and files present.")
    traceback.print_exc()
    raise

# Load gloss_map
if not os.path.exists(GLOSS_MAP_PATH):
    print(f"Warning: {GLOSS_MAP_PATH} not found — make sure you created gloss_map.json")
    gloss_map = {}
else:
    with open(GLOSS_MAP_PATH, "r") as f:
        gloss_map = json.load(f)

print("Models loaded. Gloss map size:", len(gloss_map))

# ======= Helper functions ========
def convert_to_wav(input_path: str, output_path: str):
    """
    Convert arbitrary audio file to mono 16k PCM WAV using ffmpeg.
    """
    ffmpeg.input(input_path).output(
        output_path,
        format="wav",
        acodec="pcm_s16le",
        ac=1,
        ar="16000"
    ).overwrite_output().run(quiet=True)
    return output_path

import re
import torch

def clean_gloss(text):
    """
    Convert raw model gloss into clean ASL gloss.
    - Fixes pronouns (x-you → YOU, pro2 → YOU, etc.)
    - Removes ASL alignment markers
    - Removes punctuation
    """
    gloss = text

    # --------------------------
    # 1. PRONOUN MAPPINGS
    # --------------------------

    # YOU (second person)
    gloss = re.sub(r'\b(x-|ix-|pro)2\b', 'YOU', gloss, flags=re.IGNORECASE)
    gloss = re.sub(r'\b(x-you|ix-you|you)\b', 'YOU', gloss, flags=re.IGNORECASE)

    # I / ME (first person)
    gloss = re.sub(r'\b(x-|ix-|pro)1\b', 'I', gloss, flags=re.IGNORECASE)
    gloss = re.sub(r'\b(x-me|ix-me|me|i)\b', 'I', gloss, flags=re.IGNORECASE)

    # HE / SHE / THEY (third person)
    gloss = re.sub(r'\b(x-|ix-|pro)3\b', 'HE-SHE', gloss, flags=re.IGNORECASE)
    gloss = re.sub(r'\b(x-him|ix-him|x-her|ix-her|him|her)\b', 'HE-SHE', gloss, flags=re.IGNORECASE)


    # --------------------------
    # 2. REMOVE ASL ARTIFACT TAGS
    # --------------------------
    asl_tags = (
        r'\b('
        r'x-\w+|'       # ex: x-boy, x-house (already handled pronouns above)
        r'ix-\w+|'      # pointing index
        r'loc|dir|rep|' # location, direction, repetition
        r'neg|q|wh|'    # question markers
        r'top|desc-|'   # topic, description
        r'cl:[\w/]+|'   # classifier labels
        r'p-\w+|'       # place/positional codes
        r'\+\+'         # repetition marker
        r')\b'
    )
    gloss = re.sub(asl_tags, ' ', gloss, flags=re.IGNORECASE)


    # --------------------------
    # 3. REMOVE PUNCTUATION
    # --------------------------
    gloss = re.sub(r'[.,!?;:"\'\-]', ' ', gloss)


    # --------------------------
    # 4. CLEAN EXTRA SPACES
    # --------------------------
    gloss = re.sub(r'\s+', ' ', gloss).strip()

    # Make all uppercase (standard GLOSS format)
    gloss = gloss.upper()

    return gloss



def get_gloss(text: str) -> str:
    """
    Runs model inference and cleans the gloss.
    """
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output_ids = gloss_model.generate(**inputs, max_length=128)

    raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return clean_gloss(raw)






import cv2
import os
import numpy as np

def concat_videos_opencv(sentence,
                                output="asl_smooth.mp4",
                                fade_duration=0.25,
                                gap_duration=0.1,
                                height=480,
                                default_fps=25,
                                verbose=True):
    """
    ASL concatenator with:
      - uniform frame size via padding
      - smooth crossfade between signs
      - optional black gap
      - OpenCV-only
    """

    words = sentence.strip().upper().split()

    # collect valid videos
    video_paths = []
    fps_candidates = []

    for w in words:
        if w not in gloss_map:
            if verbose: print(f"[Missing] {w}")
            continue

        p = gloss_map[w]
        if not os.path.exists(p):
            if verbose: print(f"[File not found] {p}")
            continue

        video_paths.append((w, p))

        cap = cv2.VideoCapture(p)
        if cap.isOpened():
            f = cap.get(cv2.CAP_PROP_FPS)
            if f and f > 1:
                fps_candidates.append(f)
        cap.release()

    if not video_paths:
        print("No valid videos.")
        return

    # use average fps
    fps = int(sum(fps_candidates) / len(fps_candidates)) if fps_candidates else default_fps
    fps=int(fps*4)
    if verbose: print("Chosen FPS:", fps)

    # PASS 1 — determine output width
    max_width = 0
    for gloss, p in video_paths:
        cap = cv2.VideoCapture(p)
        ret, frame = cap.read()
        cap.release()
        if not ret: continue

        h, w = frame.shape[:2]
        new_w = int((height / h) * w)
        max_width = max(max_width, new_w)

    if verbose: print("Output width:", max_width)

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, fps, (max_width, height))

    def pad_frame(frame):
        """Resize → pad frame to (max_width, height)."""
        h0, w0 = frame.shape[:2]
        frame = cv2.resize(frame, (int((height / h0) * w0), height))
        h0, w0 = frame.shape[:2]

        if w0 < max_width:
            pad_left = (max_width - w0) // 2
            pad_right = max_width - w0 - pad_left
            frame = cv2.copyMakeBorder(frame, 0, 0, pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            frame = frame[:, (w0 - max_width)//2:(w0 - max_width)//2 + max_width]

        return frame

    total_frames = 0
    prev_last_frame = None
    fade_frames = int(fade_duration * fps)
    gap_frames = int(gap_duration * fps)

    # PASS 2 — write all videos with smooth transitions
    for idx, (gloss, p) in enumerate(video_paths):
        cap = cv2.VideoCapture(p)
        if verbose: print(f"[Writing] {gloss}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(pad_frame(frame))
        cap.release()

        if len(frames) == 0: continue

        # If NOT the first clip → fade transition from previous
        if prev_last_frame is not None:
            for i in range(fade_frames):
                alpha = 1 - (i / fade_frames)
                blended = (alpha * prev_last_frame + (1 - alpha) * frames[0]).astype(np.uint8)
                out.write(blended)
                total_frames += 1

            # optional black gap
            black = np.zeros((height, max_width, 3), dtype=np.uint8)
            for _ in range(gap_frames):
                out.write(black)

        # write full current clip
        for fr in frames:
            out.write(fr)
            total_frames += 1

        prev_last_frame = frames[-1]

    out.release()

    print(f"[Saved] {output} ({total_frames} frames written)")













# def concat_videos_opencv(sentence: str,
#                          output: str,
#                          fade_duration=0.25,
#                          gap_duration=0.1,
#                          height=480,
#                          default_fps=25,
#                          verbose=False):
#     """
#     Concatenate videos using the same logic you used earlier.
#     Returns the output path or raises.
#     """
#     words = sentence.strip().upper().split()
#     video_paths = []
#     fps_candidates = []

#     for w in words:
#         if w not in gloss_map:
#             if verbose: print(f"[Missing] {w}")
#             continue
#         p = gloss_map[w]
#         if not os.path.exists(p):
#             if verbose: print(f"[File not found] {p}")
#             continue
#         video_paths.append((w, p))
#         cap = cv2.VideoCapture(p)
#         if cap.isOpened():
#             f = cap.get(cv2.CAP_PROP_FPS)
#             if f and f > 1:
#                 fps_candidates.append(f)
#         cap.release()

#     if not video_paths:
#         raise RuntimeError("No valid sign videos found for sentence.")

#     fps = int(sum(fps_candidates) / len(fps_candidates)) if fps_candidates else default_fps
#     if verbose: print("Chosen FPS:", fps)

#     # PASS 1 — compute max width
#     max_width = 0
#     for gloss, p in video_paths:
#         cap = cv2.VideoCapture(p)
#         ret, frame = cap.read()
#         cap.release()
#         if not ret: continue
#         h, w = frame.shape[:2]
#         new_w = int((height / h) * w)
#         max_width = max(max_width, new_w)

#     if verbose: print("Output width:", max_width)

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(output, fourcc, fps, (max_width, height))

#     def pad_frame(frame):
#         h0, w0 = frame.shape[:2]
#         frame = cv2.resize(frame, (int((height / h0) * w0), height))
#         h0, w0 = frame.shape[:2]
#         if w0 < max_width:
#             pad_left = (max_width - w0) // 2
#             pad_right = max_width - w0 - pad_left
#             frame = cv2.copyMakeBorder(frame, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#         else:
#             frame = frame[:, (w0 - max_width)//2:(w0 - max_width)//2 + max_width]
#         return frame

#     total_frames = 0
#     prev_last_frame = None
#     fade_frames = max(1, int(fade_duration * fps))
#     gap_frames = int(gap_duration * fps)

#     for idx, (gloss, p) in enumerate(video_paths):
#         cap = cv2.VideoCapture(p)
#         if verbose: print(f"[Writing] {gloss} from {p}")
#         frames = []
#         while True:
#             ret, frame = cap.read()
#             if not ret: break
#             frames.append(pad_frame(frame))
#         cap.release()

#         if len(frames) == 0:
#             continue

#         # fade transition
#         if prev_last_frame is not None:
#             for i in range(fade_frames):
#                 alpha = 1 - (i / fade_frames)
#                 blended = (alpha * prev_last_frame + (1 - alpha) * frames[0]).astype(np.uint8)
#                 out.write(blended)
#                 total_frames += 1
#             # optional black gap
#             black = np.zeros((height, max_width, 3), dtype=np.uint8)
#             for _ in range(gap_frames):
#                 out.write(black)
#         # write frames
#         for fr in frames:
#             out.write(fr)
#             total_frames += 1

#         prev_last_frame = frames[-1]

#     out.release()
#     if verbose: print(f"[Saved] {output} ({total_frames} frames written)")
#     return output

def safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

# ======= Endpoints ========

@app.get("/")
def root():
    return {"message": "Speech→Gloss→Sign API running"}

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file upload, runs Whisper -> Gloss model -> concatenates sign video,
    returns JSON with text, gloss and a URL to download/play the generated video.
    """
    # create unique output name
    job_id = uuid.uuid4().hex
    temp_in = TEMP_DIR / f"{job_id}_{file.filename}"
    out_video = VIDEO_OUTPUT_DIR / f"{job_id}.mp4"

    # save uploaded file
    try:
        with open(temp_in, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        # convert to WAV (whisper expects readable audio)
        wav_path = TEMP_DIR / f"{job_id}.wav"
        convert_to_wav(str(temp_in), str(wav_path))

        # transcribe with whisper
        transcription = whisper_model.transcribe(str(wav_path))
        text = transcription.get("text", "").strip()
        if not text:
            raise RuntimeError("Whisper did not return any text.")

        # generate gloss
        gloss = get_gloss(text)

        # generate concatenated sign video
        concat_videos_opencv(gloss, output=str(out_video), verbose=False)

        # success → return URLs
        video_url = f"/video/{out_video.name}"
        return JSONResponse({"text": text, "gloss": gloss, "video_url": video_url})

    except Exception as e:
        tb = traceback.format_exc()
        print("ERROR during /transcribe/:", e)
        print(tb)
        # cleanup
        safe_remove(str(out_video))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # cleanup temp inputs
        safe_remove(str(temp_in))
        safe_remove(str(TEMP_DIR / f"{job_id}.wav"))

@app.get("/video/{filename}")
def serve_video(filename: str):
    """
    Serves files from generated_videos directory.
    """
    path = VIDEO_OUTPUT_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    # use FileResponse so browser can seek / start playback
    return FileResponse(path, media_type="video/mp4", filename=filename)

# Optional: list generated videos (for debugging)
@app.get("/videos/")
def list_videos():
    files = [f.name for f in VIDEO_OUTPUT_DIR.glob("*.mp4")]
    return {"videos": files}

# ======= Run server ========
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
