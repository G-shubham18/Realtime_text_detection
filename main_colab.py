"""
Google Colab Real-time 'Thank You' Board Detection Pipeline
Optimized for Colab GPU (T4 / V100 / L4 / A100) with Ollama + Qwen3-VL

Supports:
  1. Video files (e.g. lecture.mp4)
  2. IP Camera streams (HTTP MJPEG / RTSP)
  3. Colab Browser Webcam (via WebRTC/JS bridge)
  4. Web Dashboard via FastAPI + Cloudflare Tunnel / Colab Port Forwarding
  5. Real-time Email & SMS alerts
"""

import argparse
import base64
import smtplib
import ssl
from email.message import EmailMessage
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from urllib.parse import urlparse, urlunparse

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

# ---------------------------------------------------------
# Default Configurations
# ---------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:2b")
CONFIG_FILE = "config.json"

PROMPT = """You are reading a photo taken in a classroom or lecture hall.

TASK
Look at the board (whiteboard or blackboard) in the image and decide whether the words "Thank you" are handwritten on it.

STEPS
1. Find the board and read all handwritten text on it.
2. Copy the handwritten text exactly as you read it into "board_text". If the board is empty, unreadable or not visible, use an empty string.
3. Decide "thank_you_present".

RULES
- Answer true only if the handwritten text reads "Thank you". Capital letters, punctuation, underlining or decoration around it ("THANK YOU", "Thank you!", "Thank-you") are fine.
- Answer false for other text such as "Thanks", "Thank", "Welcome", equations or notes, even if it looks similar.
- Ignore printed or typed text, slides, projector screens, posters and signs. Only handwritten board text counts.
- Do not guess. If the writing is partly erased, blocked by a person, or too unclear to read, answer false.
- "confidence" is a number from 0 to 1 for how sure you are of your answer.

OUTPUT
Reply with one JSON object only. No text outside the JSON and no code fences.
{"board_text": "<text exactly as written, or empty>", "thank_you_present": true or false, "confidence": 0.0 to 1.0, "reason": "<one short sentence>"}"""

cfg = {}
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"[config] Loaded configuration from {CONFIG_FILE}", flush=True)
    except Exception as e:
        print(f"[config] Warning: Failed to parse {CONFIG_FILE}: {e}", flush=True)

cfg_camera = cfg.get("camera", {}) if isinstance(cfg.get("camera"), dict) else {}
cfg_email = cfg.get("email", {}) if isinstance(cfg.get("email"), dict) else {}
cfg_sms = cfg.get("sms_twilio", {}) if isinstance(cfg.get("sms_twilio"), dict) else {}
cfg_det = cfg.get("detection", {}) if isinstance(cfg.get("detection"), dict) else {}

default_source = str(cfg_camera.get("source", "0"))
default_interval = float(cfg_det.get("interval_seconds", 3.0))
default_cooldown = float(cfg_det.get("cooldown_seconds", 30.0))

p = argparse.ArgumentParser(description="Thank You Detection Pipeline for Google Colab")
p.add_argument("--source", default=None, help="stream URL, video file, or 0 (webcam bridge)")
p.add_argument("--port", type=int, default=8000, help="FastAPI port")
p.add_argument("--interval", type=float, default=default_interval, help="seconds between VLM checks")
p.add_argument("--cooldown", type=float, default=default_cooldown, help="seconds between alerts")
p.add_argument("--host", default="0.0.0.0")
p.add_argument("--model", default=MODEL, help=f"Ollama model name (default: {MODEL})")
p.add_argument("--ollama_url", default=OLLAMA_URL, help=f"Ollama API url (default: {OLLAMA_URL})")

# Allow running from jupyter notebook without error on -f args
if any("ipykernel" in arg or "colab" in arg for arg in sys.argv):
    a, _ = p.parse_known_args()
else:
    a = p.parse_args()

MODEL = a.model
OLLAMA_URL = a.ollama_url
source_val = a.source if a.source is not None else default_source
SRC = 0 if str(source_val) == "0" else source_val

ALERT_EMAIL = os.environ.get("ALERT_EMAIL") or cfg_email.get("receiver_email") or ""
ALERT_PHONE = os.environ.get("ALERT_PHONE") or cfg_sms.get("receiver_phone") or ""
SMTP_USER = os.environ.get("SMTP_USER") or cfg_email.get("sender_email") or ""
SMTP_PASS = os.environ.get("SMTP_PASS") or cfg_email.get("sender_app_password") or ""
TWILIO_SID = os.environ.get("TWILIO_SID") or cfg_sms.get("account_sid") or ""
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN") or cfg_sms.get("auth_token") or ""
TWILIO_FROM = os.environ.get("TWILIO_FROM") or cfg_sms.get("from_number") or ""

print("==================================================", flush=True)
print(" 🎓 Thank You Real-Time Detector (Google Colab)")
print("==================================================", flush=True)
print(f"[*] Video Source    : {SRC}", flush=True)
print(f"[*] Ollama Model    : {MODEL}", flush=True)
print(f"[*] Ollama Endpoint : {OLLAMA_URL}", flush=True)
print(f"[*] Target Receiver : {ALERT_EMAIL if ALERT_EMAIL else '(not set)'}", flush=True)
print(f"[*] Sender Email    : {SMTP_USER if SMTP_USER else '(not set - email alerts skipped)'}", flush=True)
print(f"[*] Cooldown        : {a.cooldown}s", flush=True)
print(f"[*] Check Interval  : {a.interval}s", flush=True)
print("==================================================", flush=True)

# ---------------------------------------------------------
# State Management
# ---------------------------------------------------------
lock = threading.Lock()
latest_frame = None
verdict = {"found": False, "confidence": 0.0, "reason": "waiting", "board_text": ""}
last_latency = 0.0
avg_latency = 0.0
alert_active = False
alert_msg = ""
recent_alerts = []
detection_logs = []
detection_seq = 0
latencies = []
last_alert_t = 0.0
last_alert_wall = 0.0
max_side = 768
degraded = False


def set_frame(fr):
    global latest_frame
    with lock:
        latest_frame = fr


def mjpeg_frames(url):
    """Manual MJPEG reader: yields decoded frames from an HTTP multipart stream."""
    with requests.get(url, stream=True, timeout=(5, 10)) as resp:
        resp.raise_for_status()
        print(f"[capture] connected: status={resp.status_code} content-type={resp.headers.get('Content-Type')}", flush=True)
        buf = bytearray()
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            buf += chunk
            while True:
                s = buf.find(b"\xff\xd8")
                if s < 0:
                    if len(buf) > 2_000_000:
                        buf.clear()
                    break
                e = buf.find(b"\xff\xd9", s + 2)
                if e < 0:
                    if s > 0:
                        del buf[:s]
                    if len(buf) > 2_000_000:
                        buf.clear()
                    break
                jpg = bytes(buf[s:e + 2])
                del buf[:e + 2]
                fr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if fr is not None:
                    yield fr


def snapshot_url(url):
    u = urlparse(url)
    path = re.sub(r"/(video|videofeed)/?$", "/shot.jpg", u.path)
    if path == u.path:
        return None
    return urlunparse((u.scheme, u.netloc, path, "", "", ""))


def read_snapshots(snap):
    errs = 0
    while True:
        try:
            r = requests.get(snap, timeout=3)
            fr = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if fr is None:
                raise ValueError("bad snapshot")
            set_frame(fr)
            errs = 0
        except Exception:
            errs += 1
            if errs >= 3:
                raise
        time.sleep(0.2)


def read_cv(src, ffmpeg=False, is_file=False):
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if ffmpeg else cv2.VideoCapture(src)
    if not is_file:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = 1.0 / fps if is_file else 0.01

        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                if is_file:
                    # Loop video file smoothly
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.1)
                    continue
                else:
                    raise RuntimeError("stream ended")
            set_frame(fr)
            if is_file:
                time.sleep(frame_delay)
    finally:
        try:
            cap.release()
        except Exception:
            pass


def capture_loop():
    fails = 0
    snapshot_mode = False
    announced = None

    # Check if SRC is a local file
    if isinstance(SRC, str) and os.path.isfile(SRC):
        print(f"[capture] Reading video file: {SRC} (looping)", flush=True)
        read_cv(SRC, ffmpeg=False, is_file=True)
        return

    while True:
        try:
            if isinstance(SRC, int) or str(SRC) == "0":
                if announced != "webcam_bridge":
                    print("[capture] Waiting for browser webcam frames (feed via set_frame)...", flush=True)
                    announced = "webcam_bridge"
                time.sleep(1)
            elif SRC.lower().startswith("rtsp://"):
                if announced != "rtsp":
                    print("[capture] RTSP mode (FFmpeg)", flush=True)
                    announced = "rtsp"
                read_cv(SRC, ffmpeg=True)
            else:
                snap = snapshot_url(SRC)
                if snapshot_mode and snap:
                    if announced != "snapshot":
                        print("[capture] snapshot mode", flush=True)
                        announced = "snapshot"
                    read_snapshots(snap)
                else:
                    if announced != "mjpeg":
                        print("[capture] MJPEG mode", flush=True)
                        announced = "mjpeg"
                    got = False
                    for fr in mjpeg_frames(SRC):
                        set_frame(fr)
                        got = True
                        fails = 0
                    if not got:
                        raise RuntimeError("connected but no JPEG frames found in stream")
                    raise RuntimeError("stream ended")
        except Exception as e:
            print(f"[capture] error: {type(e).__name__}: {e}", flush=True)
            fails += 1
            if not isinstance(SRC, int) and not str(SRC).startswith("rtsp://") and fails >= 3 and snapshot_url(str(SRC)):
                snapshot_mode = True
        print("[capture] reconnecting in 2s...", flush=True)
        time.sleep(2)


def resize_max_side(img, m):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= m:
        return img
    f = m / float(s)
    return cv2.resize(img, (int(w * f), int(h * f)))


THANK_RE = re.compile(r"thank\s*-?\s*you", re.I)


def parse_verdict(text):
    try:
        t = text.strip()
        t = re.sub(r"^```(?:json)?", "", t.strip()).strip()
        t = re.sub(r"```$", "", t.strip()).strip()
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            t = t[i:j + 1]
        d = json.loads(t)
        board = str(d.get("board_text", ""))[:200]
        claimed = bool(d.get("thank_you_present", False))
        found = claimed and bool(THANK_RE.search(board))
        return {
            "found": found,
            "confidence": float(d.get("confidence", 0) or 0),
            "reason": str(d.get("reason", ""))[:200],
            "board_text": board
        }
    except Exception:
        return {"found": False, "confidence": 0.0, "reason": "parse-fail", "board_text": ""}


def call_vlm(b64imgs):
    global last_latency, avg_latency, max_side, degraded
    t0 = time.time()
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "stream": False,
            "keep_alive": "30m",
            "format": "json",
            "options": {"temperature": 0, "num_predict": 500},
            "messages": [{"role": "user", "content": PROMPT, "images": b64imgs}]
        }, timeout=120)
        dt = time.time() - t0
        msg = r.json().get("message", {})
        content = msg.get("content", "")
        v = parse_verdict(content)
        if v.get("reason") == "parse-fail" and msg.get("thinking"):
            v_think = parse_verdict(msg.get("thinking"))
            if v_think.get("reason") != "parse-fail":
                v = v_think
    except Exception as e:
        dt = time.time() - t0
        v = {"found": False, "confidence": 0.0, "reason": f"vlm-error {e}"[:200], "board_text": ""}
    last_latency = dt
    latencies.append(dt)
    if len(latencies) > 20:
        latencies.pop(0)
    avg_latency = sum(latencies) / len(latencies)
    if not degraded and avg_latency > 6.0:
        degraded = True
        max_side = 640
        print(f"[AUTO-TUNE] avg latency {avg_latency:.1f}s > 6s: reducing image size to 640px", flush=True)
    return v, dt


def send_email_async(log_entry, seq, v, frames_jpg):
    now = datetime.now()
    full_date = now.strftime("%A, %d %B %Y")
    time_12h = now.strftime("%I:%M:%S %p")
    short_date = now.strftime("%d-%b-%Y")
    now_str = log_entry["time"]
    board_text = v["board_text"]
    conf = v["confidence"]

    if not (SMTP_USER and SMTP_PASS and ALERT_EMAIL):
        ack = "⚠️ Skipped (SMTP credentials not set)"
        with lock:
            log_entry["status"] = "NO_CREDS"
            log_entry["ack"] = ack
            log_entry["badge"] = "yellow"
        print(f"[{now_str}] [DETECTED #{seq}] \"{board_text}\" (conf: {int(conf*100)}%) | [EMAIL: {ack}]", flush=True)
        return

    try:
        m = EmailMessage()
        m["From"] = SMTP_USER
        m["To"] = ALERT_EMAIL
        m["Subject"] = f"🎓 Lecture Completed - 'Thank You' Detected at {time_12h} ({short_date})"

        plain_text = f"""LECTURE COMPLETED ALERT

A handwritten 'Thank you' has been detected on the classroom board.

DETECTION DETAILS:
--------------------------------------------------
📅 Date          : {full_date}
⏰ Time          : {time_12h}
📝 Detected Text : "{board_text}"
🎯 AI Confidence : {int(conf*100)}% Match
🔢 Event No.     : #{seq}
--------------------------------------------------

Captured snapshot frame from the camera is attached below as frame_0.jpg.
"""
        m.set_content(plain_text)

        html_body = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; }}
  .container {{ max-width: 560px; margin: 0 auto; background: #ffffff; border-radius: 8px; overflow: hidden; border: 1px solid #e1e4e8; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
  .header {{ background: #0a7d2c; padding: 22px; text-align: center; color: #ffffff; }}
  .header h1 {{ margin: 0; font-size: 22px; font-weight: 700; letter-spacing: 0.5px; }}
  .header p {{ margin: 6px 0 0; font-size: 14px; opacity: 0.92; }}
  .content {{ padding: 24px; color: #24292e; }}
  .card-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; background: #fafbfc; border-radius: 6px; border: 1px solid #eaecef; }}
  .card-table td {{ padding: 12px 14px; border-bottom: 1px solid #eaecef; font-size: 14px; }}
  .card-table tr:last-child td {{ border-bottom: none; }}
  .card-table td.label {{ font-weight: 600; color: #586069; width: 38%; }}
  .card-table td.value {{ color: #24292e; }}
  .text-highlight {{ color: #0366d6; font-size: 16px; font-weight: bold; }}
  .badge {{ display: inline-block; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 12px; background: #dcffe4; color: #1a7f37; border: 1px solid #a2f2b9; }}
  .footer {{ padding: 16px; text-align: center; font-size: 12px; color: #6a737d; background: #f6f8fa; border-top: 1px solid #eaecef; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🎓 Lecture Completed</h1>
    <p>Handwritten &quot;Thank you&quot; detected on board</p>
  </div>
  <div class="content">
    <table class="card-table">
      <tr>
        <td class="label">📅 Date</td>
        <td class="value"><strong>{full_date}</strong></td>
      </tr>
      <tr>
        <td class="label">⏰ Time</td>
        <td class="value"><strong>{time_12h}</strong></td>
      </tr>
      <tr>
        <td class="label">📝 Recognized Text</td>
        <td class="value"><span class="text-highlight">&quot;{board_text}&quot;</span></td>
      </tr>
      <tr>
        <td class="label">🎯 AI Confidence</td>
        <td class="value"><span class="badge">{int(conf*100)}% Match</span></td>
      </tr>
      <tr>
        <td class="label">🔢 Event Sequence</td>
        <td class="value">#{seq}</td>
      </tr>
    </table>
    <p style="margin-top: 20px; font-size: 13px; color: #586069; text-align: center;">
      📸 <em>Snapshot photo from the camera is attached below as <strong>frame_0.jpg</strong>.</em>
    </p>
  </div>
  <div class="footer">
    Sent automatically by Thank You Realtime Detector (Google Colab)
  </div>
</div>
</body>
</html>"""
        m.add_alternative(html_body, subtype="html")

        for i, jb in enumerate(frames_jpg):
            m.add_attachment(jb, maintype="image", subtype="jpeg", filename=f"frame_{i}.jpg")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=20) as srv:
            srv.login(SMTP_USER, SMTP_PASS)
            srv.send_message(m)

        ack_t = datetime.now().strftime("%I:%M:%S %p")
        ack = f"✅ EMAIL SENT to {ALERT_EMAIL} ({ack_t})"
        with lock:
            log_entry["status"] = "SENT"
            log_entry["ack"] = ack
            log_entry["badge"] = "green"
        print(f"[{now_str}] [DETECTED #{seq}] \"{board_text}\" (conf: {int(conf*100)}%) | [EMAIL ACK: {ack}]", flush=True)
    except Exception as e:
        ack = f"❌ FAILED: {str(e)[:45]}"
        with lock:
            log_entry["status"] = "FAILED"
            log_entry["ack"] = ack
            log_entry["badge"] = "red"
        print(f"[{now_str}] [DETECTED #{seq}] \"{board_text}\" (conf: {int(conf*100)}%) | [EMAIL ACK: {ack}]", flush=True)


def send_sms(body):
    if not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM and ALERT_PHONE):
        return
    try:
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
            auth=(TWILIO_SID, TWILIO_TOKEN),
            data={"From": TWILIO_FROM, "To": ALERT_PHONE, "Body": body[:300]},
            timeout=20)
        if r.status_code in (200, 201):
            print(f"[notify] SMS sent to {ALERT_PHONE}", flush=True)
    except Exception:
        pass


def do_alert(frames_jpg, v, dt, seq, now_str):
    global last_alert_t, last_alert_wall, alert_active, alert_msg
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join("alerts", ts)
    os.makedirs(d, exist_ok=True)
    for i, jb in enumerate(frames_jpg):
        with open(os.path.join(d, f"frame_{i}.jpg"), "wb") as f:
            f.write(jb)
    info = {"timestamp": ts, "verdict": v, "latency": dt}
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(info, f, indent=2)
    last_alert_t = time.time()
    last_alert_wall = time.time()
    alert_active = True
    alert_msg = f"LECTURE COMPLETED conf={v['confidence']:.2f} board='{v['board_text']}'"
    recent_alerts.insert(0, {"dir": d, "timestamp": ts, **info})
    del recent_alerts[10:]

    log_entry = {
        "seq": seq,
        "time": now_str,
        "board_text": v["board_text"],
        "confidence": round(v["confidence"], 2),
        "status": "SENDING",
        "ack": f"⏳ Sending email to {ALERT_EMAIL}...",
        "badge": "blue"
    }
    with lock:
        detection_logs.insert(0, log_entry)
        if len(detection_logs) > 50:
            detection_logs.pop()

    threading.Thread(target=send_email_async, args=(log_entry, seq, v, frames_jpg), daemon=True).start()
    sms_body = f"Lecture completed. 'Thank you' was detected on the board at {now_str}."
    threading.Thread(target=send_sms, args=(sms_body,), daemon=True).start()


def sampler_loop():
    global verdict, alert_active, detection_seq
    last_check = 0.0
    while True:
        time.sleep(0.2)
        with lock:
            fr = None if latest_frame is None else latest_frame.copy()
            if alert_active and time.time() - last_alert_wall > 8:
                alert_active = False
        if fr is None or time.time() - last_check < a.interval:
            continue
        r = resize_max_side(fr, max_side)
        ok, enc = cv2.imencode(".jpg", r, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            last_check = time.time()
            continue
        jb = enc.tobytes()
        v, dt = call_vlm([base64.b64encode(jb).decode("utf-8")])
        last_check = time.time()
        with lock:
            verdict = v

        is_positive = bool(v["found"]) and float(v.get("confidence", 0)) >= 0.6
        if is_positive:
            detection_seq += 1
            seq = detection_seq
            now_str = datetime.now().strftime("%H:%M:%S")
            elapsed = time.time() - last_alert_t
            if last_alert_t > 0 and elapsed < a.cooldown:
                remaining = int(a.cooldown - elapsed)
                ack = f"⏳ In Cooldown ({remaining}s remaining)"
                print(f"[{now_str}] [DETECTED #{seq}] \"{v['board_text']}\" (conf: {int(v['confidence']*100)}%) | [EMAIL: ⏳ SKIPPED - Cooldown active ({remaining}s remaining)]", flush=True)
                with lock:
                    detection_logs.insert(0, {
                        "seq": seq,
                        "time": now_str,
                        "board_text": v["board_text"],
                        "confidence": round(v["confidence"], 2),
                        "status": "COOLDOWN",
                        "ack": ack,
                        "badge": "yellow"
                    })
                    if len(detection_logs) > 50:
                        detection_logs.pop()
            else:
                do_alert([jb], v, dt, seq, now_str)


# ---------------------------------------------------------
# FastAPI Dashboard & Stream Endpoints
# ---------------------------------------------------------
app = FastAPI()

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Lecture End Detection - Colab Dashboard</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }
    .container { max-width: 950px; margin: 0 auto; }
    h2 { margin: 0 0 12px 0; color: #58a6ff; font-weight: 600; }
    .banner { display: none; background: #238636; color: #fff; font-size: 22px; font-weight: bold; padding: 12px; margin: 10px 0; border-radius: 6px; text-align: center; animation: pulse 0.8s infinite alternate; }
    @keyframes pulse { to { opacity: .7; } }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 14px; margin: 12px 0; }
    .stream-container { text-align: center; background: #000; border-radius: 6px; overflow: hidden; border: 1px solid #30363d; margin-bottom: 12px; }
    .stream-container img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
    .status-bar { display: flex; flex-wrap: wrap; gap: 12px; font-family: monospace; font-size: 13px; color: #8b949e; align-items: center; }
    .status-item { background: #21262d; padding: 5px 12px; border-radius: 4px; border: 1px solid #30363d; }
    .badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
    .badge-green { background: #238636; color: #ffffff; }
    .badge-yellow { background: #9e6a03; color: #ffffff; }
    .badge-blue { background: #1f6feb; color: #ffffff; }
    .badge-red { background: #da3633; color: #ffffff; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }
    th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid #21262d; }
    th { background: #0d1117; color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
    tr:hover td { background: #21262d; }
    .log-box { max-height: 380px; overflow-y: auto; border: 1px solid #30363d; border-radius: 4px; }
    .log-header { display: flex; justify-content: space-between; align-items: center; }
    .log-header h3 { margin: 0; font-size: 16px; color: #f0f6fc; }
  </style>
</head>
<body>
<div class="container">
  <h2>🎓 Lecture End Detection (Google Colab Cloud)</h2>
  <div id="b" class="banner">🎉 LECTURE COMPLETED - 'THANK YOU' DETECTED!</div>

  <div class="stream-container">
    <img src="/stream" alt="Live Stream">
  </div>

  <div class="card">
    <div class="status-bar">
      <div class="status-item">Verdict: <span id="v_found">waiting...</span></div>
      <div class="status-item">Board Text: <span id="v_text">"-"</span></div>
      <div class="status-item">Inference Latency: <span id="v_lat">0.0s</span></div>
      <div class="status-item">Email Cooldown: <span id="v_cd">Ready</span></div>
    </div>
  </div>

  <div class="card">
    <div class="log-header">
      <h3>Detection & Email ACK Logs</h3>
      <span style="font-size: 12px; color: #8b949e;" id="log_count">0 events</span>
    </div>
    <div class="log-box">
      <table>
        <thead>
          <tr>
            <th style="width: 45px;">#</th>
            <th style="width: 85px;">Time</th>
            <th>Detected Text</th>
            <th style="width: 60px;">Conf</th>
            <th style="width: 100px;">Detection</th>
            <th>Email Status / ACK</th>
          </tr>
        </thead>
        <tbody id="log_body">
          <tr><td colspan="6" style="text-align: center; color: #8b949e; padding: 20px;">No detection events yet. Point camera at "Thank you".</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
// Web Audio Beep when alert fires
let audioCtx = null;
function playBeep() {
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.frequency.value = 880; // A5 note
    gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
    osc.start();
    setTimeout(() => { osc.stop(); }, 400);
  } catch (e) {}
}

let wasAlerting = false;

setInterval(async () => {
  try {
    const res = await fetch('/status');
    const j = await res.json();

    if (j.alert && !wasAlerting) {
      playBeep();
    }
    wasAlerting = !!j.alert;

    document.getElementById('b').style.display = j.alert ? 'block' : 'none';
    document.getElementById('v_found').innerHTML = j.verdict.found ? '<span style="color:#3fb950">Thank you (' + Math.round(j.verdict.confidence * 100) + '%)</span>' : '<span style="color:#8b949e">None</span>';
    document.getElementById('v_text').innerText = j.verdict.board_text ? '"' + j.verdict.board_text + '"' : '""';
    document.getElementById('v_lat').innerText = j.last_latency.toFixed(2) + 's (avg ' + j.avg_latency.toFixed(2) + 's)';

    if (j.cooldown_remaining > 0) {
      document.getElementById('v_cd').innerHTML = '<span style="color:#d29922">⏳ ' + j.cooldown_remaining + 's remaining</span>';
    } else {
      document.getElementById('v_cd').innerHTML = '<span style="color:#3fb950">✅ Ready</span>';
    }

    if (j.logs && j.logs.length > 0) {
      document.getElementById('log_count').innerText = j.logs.length + ' events';
      const rows = j.logs.map(x => `
        <tr>
          <td>#${x.seq}</td>
          <td>${x.time}</td>
          <td style="color:#58a6ff; font-weight: 500;">"${x.board_text}"</td>
          <td>${Math.round(x.confidence * 100)}%</td>
          <td><span class="badge badge-green">🎯 DETECTED</span></td>
          <td><span class="badge badge-${x.badge}">${x.ack}</span></td>
        </tr>
      `).join('');
      document.getElementById('log_body').innerHTML = rows;
    }
  } catch (e) {}
}, 1000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/status")
def status():
    with lock:
        v = dict(verdict)
        al = list(recent_alerts)
        dl = [dict(x) for x in detection_logs]
        ll, alv, ab, am = last_latency, avg_latency, alert_active, alert_msg
        elapsed = time.time() - last_alert_t if last_alert_t > 0 else 9999
        cd_rem = max(0, int(a.cooldown - elapsed))
    return {
        "verdict": v,
        "last_latency": ll,
        "avg_latency": alv,
        "alert": ab,
        "alert_msg": am,
        "alerts": al,
        "logs": dl,
        "cooldown_remaining": cd_rem,
        "cooldown_total": a.cooldown
    }


def ascii_text(t, n):
    return str(t).encode("ascii", "replace").decode()[:n]


def mjpeg():
    while True:
        with lock:
            fr = None if latest_frame is None else latest_frame.copy()
            v = dict(verdict)
            ll, ab = last_latency, alert_active
        if fr is None:
            time.sleep(0.1)
            continue
        h, w = fr.shape[:2]
        s = f"thank_you={v['found']} conf={v['confidence']:.2f} lat={ll:.2f}s"
        cv2.putText(fr, s, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if v["found"] else (255, 255, 255), 2)
        cv2.putText(fr, "board: " + ascii_text(v.get("board_text", ""), 50), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if ab:
            cv2.rectangle(fr, (0, 0), (w, 90), (0, 140, 0), -1)
            cv2.putText(fr, "LECTURE COMPLETED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        ok, enc = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + enc.tobytes() + b"\r\n"
        time.sleep(0.1)


@app.get("/stream")
def stream():
    return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


# Helper endpoint for browser webcam upload
@app.post("/feed_frame")
async def feed_frame_api(data: dict):
    """Allows uploading frames from browser JS directly."""
    try:
        b64 = data.get("image", "")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        fr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if fr is not None:
            set_frame(fr)
            return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    return {"status": "invalid frame"}


def start_pipeline():
    """Entry point to start threads when imported into Jupyter/Colab notebook."""
    t_cap = threading.Thread(target=capture_loop, daemon=True)
    t_cap.start()
    t_sam = threading.Thread(target=sampler_loop, daemon=True)
    t_sam.start()
    return t_cap, t_sam


if __name__ == "__main__":
    start_pipeline()
    uvicorn.run(app, host=a.host, port=a.port)
