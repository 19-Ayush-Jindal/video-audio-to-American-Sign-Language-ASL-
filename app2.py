# # # # GET YOUTUBE AUDIO
# # # import yt_dlp

# # # def download_audio(url, output="audio.mp3"):
# # #     ydl_opts = {
# # #         "format": "bestaudio/best",
# # #         "outtmpl": output,
# # #         "quiet": True,
# # #         "postprocessors": [{
# # #             "key": "FFmpegExtractAudio",
# # #             "preferredcodec": "mp3",
# # #             "preferredquality": "192",
# # #         }],
# # #     }
# # #     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
# # #         ydl.download([url])

# # #     return output
# # # #  SPLIT IN 5 SEC CHUNKS
# # # from pydub import AudioSegment

# # # def split_audio_chunks(path, chunk_ms=5000):
# # #     audio = AudioSegment.from_file(path)
# # #     chunks = []

# # #     for i in range(0, len(audio), chunk_ms):
# # #         chunk = audio[i:i+chunk_ms]
# # #         filename = f"chunk_{i//chunk_ms}.wav"
# # #         chunk.export(filename, format="wav")
# # #         chunks.append(filename)

# # #     return chunks

# # # from fastapi import WebSocket
# # # app = FastAPI()
# # # @app.websocket("/stream_asl")
# # # async def stream_asl(ws: WebSocket):
# # #     await ws.accept()

# # #     url = await ws.receive_text()     # receive YouTube URL
# # #     audio_path = download_audio(url)
# # #     chunks = split_audio_chunks(audio_path)

# # #     for chunk_path in chunks:

# # #     # 1. Upload each chunk
# # #         files = {"file": open(chunk_path, "rb")}
# # #         response = requests.post("http://localhost:8000/transcribe/", files=files)
# # #         data = response.json()

# # #         print("Chunk gloss:", data["gloss"])
# # #         print("Video URL:", data["video_url"])

# # #     # 2. Download the generated video
# # #         video_url = "http://localhost:8000" + data["video_url"]
# # #         video_bytes = requests.get(video_url).content

# # #     # 3. Send video back to FE over WebSocket
# # #         await ws.send_bytes(video_bytes)

# # # GET YOUTUBE AUDIO
# # import sys, types

# # # Fake pyaudioop for Python 3.13 (safe patch)
# # sys.modules["pyaudioop"] = types.ModuleType("pyaudioop")

# # import yt_dlp
# # import requests
# # from fastapi import FastAPI, WebSocket
# # import sys
# # sys.modules["pyaudioop"] = None

# # from pydub import AudioSegment

# # app = FastAPI()


# # # ----------------------------------------
# # # DOWNLOAD AUDIO FROM YOUTUBE
# # # ----------------------------------------
# # def download_audio(url, output="audio.mp3"):
# #     ydl_opts = {
# #         "format": "bestaudio/best",
# #         "outtmpl": output,
# #         "quiet": True,
# #         "postprocessors": [{
# #             "key": "FFmpegExtractAudio",
# #             "preferredcodec": "mp3",
# #             "preferredquality": "192",
# #         }],
# #     }
# #     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
# #         ydl.download([url])

# #     return output


# # # ----------------------------------------
# # # SPLIT AUDIO INTO 5 SECOND CHUNKS
# # # ----------------------------------------
# # def split_audio_chunks(path, chunk_ms=5000):
# #     audio = AudioSegment.from_file(path)
# #     chunks = []

# #     for i in range(0, len(audio), chunk_ms):
# #         chunk = audio[i:i + chunk_ms]
# #         filename = f"chunk_{i//chunk_ms}.wav"
# #         chunk.export(filename, format="wav")
# #         chunks.append(filename)

# #     return chunks


# # # ----------------------------------------
# # # STREAM ASL VIDEOS CHUNK BY CHUNK
# # # ----------------------------------------
# # @app.websocket("/stream_asl")
# # async def stream_asl(ws: WebSocket):
# #     await ws.accept()

# #     # Receive YouTube URL from frontend
# #     youtube_url = await ws.receive_text()

# #     # 1. Download audio from YouTube
# #     audio_path = download_audio(youtube_url)

# #     # 2. Split into small chunks
# #     chunks = split_audio_chunks(audio_path)

# #     # 3. Process each chunk
# #     for chunk_path in chunks:

# #         # Send audio chunk to /transcribe/
# #         with open(chunk_path, "rb") as f:
# #             files = {"file": f}
# #             response = requests.post(
# #                 "http://localhost:8000/transcribe/",
# #                 files=files
# #             )

# #         data = response.json()

# #         print("\n🎧 Chunk Gloss:", data.get("gloss"))
# #         print("🎞 Video Path:", data.get("video_url"))

# #         # 4. Download generated video
# #         video_url = "http://localhost:8000" + data["video_url"]
# #         video_bytes = requests.get(video_url).content

# #         # 5. Send video bytes to frontend WebSocket
# #         await ws.send_bytes(video_bytes)
# # ---------- PYTHON 3.13 FIX ----------
# # Pydub imports "pyaudioop", which is removed in Python 3.13
# # We create a fake empty module so Pydub does not crash.
# import sys, types
# import numpy as np

# fake = types.ModuleType("pyaudioop")

# def mul(fragment, width, factor):
#     arr = np.frombuffer(fragment, dtype="<i2")  # 16-bit audio
#     arr = (arr * factor).astype("<i2")
#     return arr.tobytes()

# def add(fragment1, fragment2, width):
#     a = np.frombuffer(fragment1, dtype="<i2")
#     b = np.frombuffer(fragment2, dtype="<i2")
#     c = np.clip(a + b, -32768, 32767).astype("<i2")
#     return c.tobytes()

# def bias(fragment, width, bias):
#     arr = np.frombuffer(fragment, dtype="<i2")
#     arr = np.clip(arr + bias, -32768, 32767).astype("<i2")
#     return arr.tobytes()

# fake.mul = mul
# fake.add = add
# fake.bias = bias

# # Register fake module
# sys.modules["pyaudioop"] = fake

# import sys, types
# sys.modules["pyaudioop"] = types.ModuleType("pyaudioop")
# # -------------------------------------

# import yt_dlp
# import requests
# from fastapi import FastAPI, WebSocket
# from pydub import AudioSegment

# app = FastAPI()


# # --------------------------------------------------------
# # DOWNLOAD AUDIO FROM YOUTUBE
# # --------------------------------------------------------
# def download_audio(url, output="audio"):
#     ydl_opts = {
#         "format": "bestaudio/best",
#         "outtmpl": output,
#         "quiet": True,
#         "postprocessors": [{
#             "key": "FFmpegExtractAudio",
#             "preferredcodec": "mp3",
#             "preferredquality": "192",
#         }],
#     }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         ydl.download([url])

#     return output


# # --------------------------------------------------------
# # SPLIT AUDIO INTO 5 SECOND CHUNKS
# # --------------------------------------------------------
# def split_audio_chunks(path, chunk_ms=5000):
#     # print("splitting")
#     audio = AudioSegment.from_file(path)
#     chunks = []

#     for i in range(0, len(audio), chunk_ms):
#         print("splitting")
#         chunk = audio[i:i + chunk_ms]
#         filename = f"chunk_{i // chunk_ms}.wav"
#         chunk.export(filename, format="wav")
#         chunks.append(filename)

#     return chunks

# import time
# # --------------------------------------------------------
# # STREAM ASL VIDEOS (CHUNK BY CHUNK)
# # --------------------------------------------------------
# @app.websocket("/stream_asl")
# async def stream_asl(ws: WebSocket):
#     await ws.accept()

#     # 1. Receive YouTube URL from frontend
#     youtube_url = await ws.receive_text()

#     # 2. Download audio from YouTube
#     download_audio(youtube_url)
#     audio_path="audio.mp3"

#     # 3. Convert into 5-sec WAV chunks
#     chunks = split_audio_chunks(audio_path)
#     # time.sleep(10)
#     print("check1")
#     # 4. Process and stream each chunk
#     for chunk_path in chunks:

#         # Send chunk to main API (/transcribe/)
#         with open(chunk_path, "rb") as f:
#             files = {"file": f}
#             response = requests.post("http://localhost:8000/transcribe/", files=files)

#         data = response.json()

#         print("\n🎧 Chunk Gloss:", data.get("gloss"))
#         print("🎞 Video Path:", data.get("video_url"))

#         # Get the generated video from API
#         video_url = "http://localhost:8000" + data["video_url"]
#         video_bytes = requests.get(video_url).content

#         # Send video bytes back to frontend via WebSocket
#         await ws.send_bytes(video_bytes)

import sys, types
sys.modules["pyaudioop"] = types.ModuleType("pyaudioop")   # Safe dummy module

import yt_dlp
import requests
import subprocess
import os
from fastapi import FastAPI, WebSocket
import asyncio

app = FastAPI()

import subprocess
import uuid

# def ensure_h264(input_path):
#     output_path = f"temp_{uuid.uuid4().hex}.mp4"

#     subprocess.run([
#         "ffmpeg", "-y",
#         "-i", input_path,
#         "-vcodec", "libx264",
#         "-acodec", "aac",
#         "-preset", "fast",
#         output_path
#     ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     return output_path

import subprocess, uuid, os

def ensure_h264(input_path):
    output_path = f"temp_{uuid.uuid4().hex}.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # If ffmpeg failed → log error
    if result.returncode != 0:
        print("❌ FFmpeg FAILED:")
        print(result.stderr.decode())

        # fallback: return original file
        if os.path.exists(input_path):
            print("⚠ Using raw file as fallback:", input_path)
            return input_path

        return None

    # If ffmpeg output does not exist → fallback
    if not os.path.exists(output_path):
        print("❌ H264 output missing, fallback to raw")
        return input_path

    return output_path


def download_audio(url, output="audio.mp3"):
    ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": "audio",
    "quiet": True,
    "nocheckcertificate": True,
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
}


    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # --- NEW: Verify audio file really exists and is not HTML ---
    if not os.path.exists(output) or os.path.getsize(output) < 1000:
        print("❌ audio.mp3 is INVALID or SABR protected.")
        # Read and print first bytes to check
        with open(output, "rb") as f:
            head = f.read(100)
            print("File header:", head)
        raise Exception("❌ Downloaded audio is invalid (SABR protection). Try different format.")

    print("Downloaded:", output)
    return output

def download_audio4(url, output="audio"):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output,
        "quiet": True,
        "extractor_args": {
            "youtube": {
                "player_client": "android"   # ★ FIX: avoid SABR
            }
        },
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # if not os.path.exists(output) or os.path.getsize(output) < 500:
    #    raise Exception("❌ Downloaded audio is empty (SABR protected). Try another video.")

    return output

def download_audio2(url, output="audio"):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output,
        "quiet": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["default"]   # 👈 Force non-Safari client
            }
        },
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # ---- Validate file ----
    # if not os.path.exists(output) or os.path.getsize(output) < 5000:
        # raise Exception("❌ Downloaded audio is empty (SABR protected). Try another video.")

    return output

# --------------------------------------------------------
# DOWNLOAD AUDIO FROM YOUTUBE (MP3)
# --------------------------------------------------------
def download_audio1(url, output="audio.mp3"):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output,
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output
def split_audio_chunks(input_path, chunk_ms=5000):

    chunk_seconds = chunk_ms // 1000
    output_files = []

    # Get duration using ffprobe
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    duration_text = result.stdout.strip()
    print("Raw ffprobe duration:", duration_text)

    if not duration_text:
        # fallback method
        print("⚠ ffprobe failed → using fallback")
        result = subprocess.run(
            [
                "ffprobe",
                "-i", input_path,
                "-show_entries", "format=duration",
                "-v", "quiet",
                "-of", "csv=p=0"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        duration_text = result.stdout.strip()

    if not duration_text:
        raise Exception("❌ Could not read audio duration. ffprobe output empty.")

    duration = float(duration_text)
    print("Duration parsed:", duration)

    # Create chunks
    index = 0
    start = 0

    while start < duration:
        out_file = f"chunk_{index}.wav"
        output_files.append(out_file)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-ss", str(start),
                "-t", str(chunk_seconds),
                out_file
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print("Created:", out_file)
        index += 1
        start += chunk_seconds

    return output_files


# --------------------------------------------------------
# SPLIT AUDIO USING FFMPEG (NO PYDUB!)
# --------------------------------------------------------
def split_audio_chunks1(input_path, chunk_ms=5000):

    chunk_seconds = chunk_ms // 1000
    output_files = []

    # Get audio duration (in seconds)
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    duration = float(result.stdout.strip())-7  # Slightly less to avoid overshoot
    print("Duration:", duration)

    # Create chunks using ffmpeg
    index = 0
    start = 0

    while start < duration:
        out_file = f"chunk_{index}.wav"
        output_files.append(out_file)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-ss", str(start),
                "-t", str(chunk_seconds),
                out_file
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print("Created:", out_file)
        index += 1
        start += chunk_seconds

    return output_files


# --------------------------------------------------------
# STREAM ASL VIDEOS OVER WEBSOCKET
# --------------------------------------------------------


import json
@app.websocket("/stream_asl")

async def stream_asl(ws: WebSocket):
    fixed_videos = []

    # is_paused = False
    # is_running = True
    await ws.accept()

    # Receive YouTube URL from frontend
    # msg= await ws.receive_text()
    # data=json.loads(msg)

    # if data["action"] == "pause":
        # is_paused = True
        # continue

    # if data["action"] == "resume":
    #     is_paused = False
    #     # continue

    # if data["action"] == "stop":
    #     is_running = False
    #     # break

    # if data["action"] == "start":
    # youtube_url = data#["youtube_url"]
    youtube_url = await ws.receive_text()

    print("Received URL:", youtube_url)

    # 1. Download audio
    audio_path = download_audio(youtube_url)
    # audio_path="audio.mp3"
    print("Downloaded:", audio_path)


    audio_path="audio.mp3"
    # 2. Split audio safely
    chunks = split_audio_chunks(audio_path)
    print("Chunks created:", chunks)

    # 3. Send each chunk to /transcribe/
    for chunk_path in chunks:
        # if not is_running:
            # break

        # while is_paused:
            # await asyncio.sleep(0.1)
            # continue
        print("Processing:", chunk_path)
        

        with open(chunk_path, "rb") as f:
            resp = requests.post(
                "http://localhost:8000/transcribe/",
                files={"file": f}
            )

        data = resp.json()
        print(data)
        gloss = data.get("gloss")
        # video_url = data.get("video_url")

        # print("Gloss:", gloss)
        # print("Video URL:", video_url)

        # # 4. Fetch video bytes
        # full_video_url = "http://localhost:8000" + video_url
        # # full_video_url =video_url

        # video_bytes = requests.get(full_video_url).content

        # # 5. Send the video back to frontend
        # await ws.send_text(full_video_url)
        video_url = data.get("video_url")
        if not video_url:
            print("❌ No video_url returned, skipping this chunk.")
            continue
        
        full_video_url = "http://localhost:8000" + video_url

# Download original video
        raw_path = "raw_chunk.mp4"
        open(raw_path, "wb").write(requests.get(full_video_url).content)

# Re-encode to H264
        fixed_path = ensure_h264(raw_path)
        # fixed_videos.append(fixed_path)
# Send *H.264 compatible* video
        if not fixed_path or not os.path.exists(fixed_path):
            print("❌ FIXED VIDEO NOT FOUND:", fixed_path)
            continue
        fixed_videos.append(fixed_path)
        await ws.send_bytes(open(fixed_path, "rb").read())
        
        # await ws.send_bytes(video_bytes)
        import time
        # time.sleep(5)
        await asyncio.sleep(0.3)  # Small delay to avoid overloading frontend
    concat_list = "concat_files.txt"
    with open(concat_list, "w") as f:
        for vid in fixed_videos:
            f.write(f"file '{vid}'\n")

    final_output = "final_output.mp4"

# Concatenate all chunks
    subprocess.run([
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", concat_list,
    "-c", "copy",
    final_output
])
    await ws.send_text("DONE")
    # After concatenation completes

    # if os.path.exists(final_output):
    # # Notify frontend that final video is ready
    #     await ws.send_json({
    #     "type": "final_video_ready"
    # })

    # # Send the final merged MP4 file
    #     with open(final_output, "rb") as f:
    #         await ws.send_bytes(f.read())
