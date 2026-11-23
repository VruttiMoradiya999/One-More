"""
One-More x REX.FITNESS STUDIO: Strict AI Pushup Coach.
Separated multi-page web application:
  - /        : Clean, high-impact Landing Page
  - /studio  : Dedicated AI Live Camera Workout Studio
  - /camera  : Dedicated Camera Setup & Calibration Guide
"""

import base64
import math
import os
import time
from collections import deque
from enum import Enum

import cv2
import numpy as np
import mediapipe as mp
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Biomechanics Constants
ELBOW_DOWN = 90
ELBOW_UP = 160
ELBOW_START = 150
BODY_MIN = 160
MAX_TILT_DEG = 32
HANDS_UNDER_SHOULDER = 0.35
MIN_VISIBILITY = 0.6
COUNTDOWN_DURATION = 10.0
READY_HOLD_SEC = 1.0
LOST_POSITION_SEC = 0.7
CONFIRM_FRAMES = 3
MIN_REP_SEC = 0.8
MAX_REP_SEC = 15.0
SMOOTH_FRAMES = 5

mp_pose = mp.solutions.pose
P = mp_pose.PoseLandmark

SIDES = {
    "left": [P.LEFT_SHOULDER, P.LEFT_ELBOW, P.LEFT_WRIST, P.LEFT_HIP, P.LEFT_ANKLE],
    "right": [P.RIGHT_SHOULDER, P.RIGHT_ELBOW, P.RIGHT_WRIST, P.RIGHT_HIP, P.RIGHT_ANKLE],
}


class AppMode(str, Enum):
    IDLE = "IDLE"
    COUNTDOWN = "COUNTDOWN"
    WORKOUT = "WORKOUT"
    STOPPED = "STOPPED"


def calc_angle(a, b, c):
    v1, v2 = a - b, c - b
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 * norm2 < 1e-7:
        return 180.0
    cos = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def choose_best_side(lm):
    scores = {s: min(lm[i].visibility for i in ids) for s, ids in SIDES.items()}
    return max(scores, key=scores.get)


def is_plank_geometry_ok(sh, wr, ank):
    dx, dy = abs(sh[0] - ank[0]), abs(sh[1] - ank[1])
    tilt = np.degrees(np.arctan2(dy, dx + 1e-9))
    body_len = np.hypot(dx, dy) + 1e-9
    horizontal = tilt <= MAX_TILT_DEG
    hands_under = abs(wr[0] - sh[0]) <= HANDS_UNDER_SHOULDER * body_len
    return bool(horizontal and hands_under)


def diagnose_hip_problem(sh, hip, ank):
    if abs(ank[0] - sh[0]) < 1e-6:
        return "Keep body straight"
    t = (hip[0] - sh[0]) / (ank[0] - sh[0])
    line_y = sh[1] + t * (ank[1] - sh[1])
    return "Hips too high" if hip[1] < line_y else "Hips sagging"


class WorkoutEngine:
    def __init__(self):
        self.mode = AppMode.IDLE
        self.countdown_start = 0.0
        self.workout_start = 0.0
        self.workout_duration = 0.0

        self.good_reps = 0
        self.bad_reps = 0

        self.state = "WAIT"
        self.side = None
        self.elbow_hist = deque(maxlen=SMOOTH_FRAMES)
        self.body_hist = deque(maxlen=SMOOTH_FRAMES)

        self.hold_start = None
        self.lost_since = None
        self.rep_start_time = 0.0
        self.body_ok = True
        self.problem = ""
        self.down_cnt = 0
        self.up_cnt = 0
        self.body_bad_cnt = 0

        self.elbow_angle = 180.0
        self.body_angle = 180.0
        self.status_msg = "Waiting to start"
        self.feedback = ""
        self.feedback_time = 0.0

        self.pose = mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def start(self):
        self.mode = AppMode.COUNTDOWN
        self.countdown_start = time.time()
        self.good_reps = 0
        self.bad_reps = 0
        self.feedback = ""
        self.reset_rep_machine()

    def stop(self):
        if self.mode == AppMode.WORKOUT:
            self.workout_duration = time.time() - self.workout_start
        else:
            self.workout_duration = 0.0
        self.mode = AppMode.STOPPED
        self.state = "WAIT"

    def reset_rep_machine(self):
        self.state = "WAIT"
        self.hold_start = None
        self.lost_since = None
        self.down_cnt = 0
        self.up_cnt = 0
        self.body_bad_cnt = 0
        self.body_ok = True
        self.problem = ""
        self.elbow_hist.clear()
        self.body_hist.clear()

    def process_frame(self, frame_bgr):
        now = time.time()
        h, w = frame_bgr.shape[:2]

        if self.mode == AppMode.COUNTDOWN:
            elapsed = now - self.countdown_start
            if elapsed >= COUNTDOWN_DURATION:
                self.mode = AppMode.WORKOUT
                self.workout_start = time.time()
                self.reset_rep_machine()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        landmarks_out = []
        valid = False
        sh = hip = ank = None

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            if self.side is None or self.state == "WAIT":
                new_side = choose_best_side(lm)
                if new_side != self.side:
                    self.side = new_side
                    self.elbow_hist.clear()
                    self.body_hist.clear()

            ids = SIDES[self.side]
            vis = min(lm[i].visibility for i in ids)

            for i in ids:
                landmarks_out.append({
                    "id": int(i),
                    "x": float(lm[i].x),
                    "y": float(lm[i].y),
                    "vis": float(lm[i].visibility)
                })

            if vis >= MIN_VISIBILITY:
                points = [np.array([lm[i].x * w, lm[i].y * h]) for i in ids]
                sh, el, wr, hip, ank = points
                self.elbow_hist.append(calc_angle(sh, el, wr))
                self.body_hist.append(calc_angle(sh, hip, ank))
                self.elbow_angle = float(np.mean(self.elbow_hist))
                self.body_angle = float(np.mean(self.body_hist))

                valid = is_plank_geometry_ok(sh, wr, ank)
                self.status_msg = "In position" if valid else "Not horizontal"
            else:
                self.status_msg = "Move back — need full body"
        else:
            self.status_msg = "No person detected"

        if self.mode == AppMode.WORKOUT:
            self.update_workout(valid, sh, hip, ank, now)

        timer_str = "00:00.0"
        countdown_rem = 0.0

        if self.mode == AppMode.COUNTDOWN:
            countdown_rem = max(0.0, COUNTDOWN_DURATION - (now - self.countdown_start))
            timer_str = f"PREP: {countdown_rem:04.1f}s"
        elif self.mode == AppMode.WORKOUT:
            elapsed = now - self.workout_start
            m, s = divmod(int(elapsed), 60)
            tenths = int((elapsed - int(elapsed)) * 10)
            timer_str = f"{m:02d}:{s:02d}.{tenths}"
        elif self.mode == AppMode.STOPPED:
            m, s = divmod(int(self.workout_duration), 60)
            tenths = int((self.workout_duration - int(self.workout_duration)) * 10)
            timer_str = f"{m:02d}:{s:02d}.{tenths}"

        total = self.good_reps + self.bad_reps
        accuracy = int(self.good_reps / total * 100) if total > 0 else 100

        return {
            "mode": self.mode.value,
            "state": self.state,
            "timer_str": timer_str,
            "countdown_rem": round(countdown_rem, 1),
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "accuracy": accuracy,
            "elbow_angle": round(self.elbow_angle, 1),
            "body_angle": round(self.body_angle, 1),
            "status_msg": self.status_msg,
            "feedback": self.feedback if (now - self.feedback_time < 2.5) else "",
            "is_good_form": bool(valid and self.body_angle >= BODY_MIN),
            "landmarks": landmarks_out,
            "side": self.side
        }

    def update_workout(self, valid, sh, hip, ank, now):
        elbow = self.elbow_angle
        body = self.body_angle

        if not valid:
            self.hold_start = None
            if self.lost_since is None:
                self.lost_since = now
            if self.state != "WAIT" and (now - self.lost_since) > LOST_POSITION_SEC:
                if self.state in ("LOWERING", "DOWN"):
                    self.feedback = "Rep cancelled: left position"
                    self.feedback_time = now
                self.reset_rep_machine()
            return

        self.lost_since = None

        if self.state == "WAIT":
            if elbow >= ELBOW_UP and body >= BODY_MIN:
                if self.hold_start is None:
                    self.hold_start = now
                if now - self.hold_start >= READY_HOLD_SEC:
                    self.state = "UP"
                    self.feedback = "READY! Start your pushups"
                    self.feedback_time = now
            else:
                self.hold_start = None

        elif self.state == "UP":
            if elbow < ELBOW_START:
                self.state = "LOWERING"
                self.rep_start_time = now
                self.body_ok = True
                self.problem = ""
                self.down_cnt = 0
                self.up_cnt = 0
                self.body_bad_cnt = 0

        elif self.state in ("LOWERING", "DOWN"):
            if body < BODY_MIN:
                self.body_bad_cnt += 1
            else:
                self.body_bad_cnt = 0

            if self.body_bad_cnt >= CONFIRM_FRAMES and self.body_ok:
                self.body_ok = False
                self.problem = diagnose_hip_problem(sh, hip, ank)

            if now - self.rep_start_time > MAX_REP_SEC:
                self.feedback = "No point: too slow"
                self.feedback_time = now
                self.state = "UP"
                self.reset_rep_machine()

            elif self.state == "LOWERING":
                self.down_cnt = (self.down_cnt + 1) if elbow <= ELBOW_DOWN else 0
                if self.down_cnt >= CONFIRM_FRAMES:
                    self.state = "DOWN"
                    self.up_cnt = 0
                else:
                    self.up_cnt = (self.up_cnt + 1) if elbow >= ELBOW_UP else 0
                    if self.up_cnt >= CONFIRM_FRAMES:
                        self.bad_reps += 1
                        self.feedback = "No point: go lower"
                        self.feedback_time = now
                        self.state = "UP"
                        self.reset_rep_machine()

            elif self.state == "DOWN":
                self.up_cnt = (self.up_cnt + 1) if elbow >= ELBOW_UP else 0
                if self.up_cnt >= CONFIRM_FRAMES:
                    problems = []
                    if not self.body_ok:
                        problems.append(self.problem or "keep body straight")
                    if now - self.rep_start_time < MIN_REP_SEC:
                        problems.append("too fast")

                    if problems:
                        self.bad_reps += 1
                        self.feedback = "No point: " + ", ".join(problems)
                    else:
                        self.good_reps += 1
                        self.feedback = f"GOOD REP! (#{self.good_reps})"

                    self.feedback_time = now
                    self.state = "UP"
                    self.reset_rep_machine()


# ============================== 2-PLAYER BATTLE ENGINE ==============================
class BattleEngine:
    def __init__(self):
        self.p1 = WorkoutEngine()
        self.p2 = WorkoutEngine()
        self.mode = AppMode.IDLE
        self.p1_ready = False
        self.p2_ready = False
        self.countdown_start = 0.0
        self.battle_start = 0.0
        self.battle_duration = 60.0  # 60s battle
        self.elapsed = 0.0
        self.winner = None  # None, "P1", "P2", "DRAW"
        self.win_reason = ""
        self.p1_dominance = 50.0  # Tug-of-war dominance (0.0 to 100.0)
        self.p2_dominance = 50.0
        self.last_p1_ping = 0.0
        self.last_p2_ping = 0.0

    def set_ready(self, player: int, ready: bool = True):
        if player == 1:
            self.p1_ready = ready
        elif player == 2:
            self.p2_ready = ready

        # When BOTH players click start/ready, begin the synchronized battle!
        if self.p1_ready and self.p2_ready and self.mode in [AppMode.IDLE, AppMode.STOPPED]:
            self.start_countdown()

    def start_countdown(self, duration=60.0):
        self.battle_duration = duration
        self.mode = AppMode.COUNTDOWN
        self.countdown_start = time.time()
        self.winner = None
        self.win_reason = ""
        self.elapsed = 0.0
        self.p1_dominance = 50.0
        self.p2_dominance = 50.0
        self.p1.start()
        self.p2.start()

    def start(self, duration=60.0):
        # Force start both
        self.p1_ready = True
        self.p2_ready = True
        self.start_countdown(duration)

    def stop(self):
        self.mode = AppMode.STOPPED
        self.p1_ready = False
        self.p2_ready = False
        self.p1.stop()
        self.p2.stop()
        self._determine_winner("MATCH STOPPED")

    def reset(self):
        self.mode = AppMode.IDLE
        self.p1_ready = False
        self.p2_ready = False
        self.p1 = WorkoutEngine()
        self.p2 = WorkoutEngine()
        self.winner = None
        self.win_reason = ""
        self.p1_dominance = 50.0
        self.p2_dominance = 50.0

    def _determine_winner(self, default_reason="TIME EXPIRED"):
        if self.p1.good_reps > self.p2.good_reps:
            self.winner = "P1"
            self.win_reason = f"PLAYER 1 WINS ({self.p1.good_reps} vs {self.p2.good_reps} REPS)"
        elif self.p2.good_reps > self.p1.good_reps:
            self.winner = "P2"
            self.win_reason = f"PLAYER 2 WINS ({self.p2.good_reps} vs {self.p1.good_reps} REPS)"
        else:
            self.winner = "DRAW"
            self.win_reason = f"DRAW MATCH ({self.p1.good_reps} vs {self.p2.good_reps} REPS)"

    def process_single_frame(self, player: int, frame_bgr):
        now = time.time()
        if player == 1:
            self.last_p1_ping = now
        else:
            self.last_p2_ping = now

        # Synchronize transition from COUNTDOWN to WORKOUT
        if self.mode == AppMode.COUNTDOWN:
            elapsed = now - self.countdown_start
            if elapsed >= COUNTDOWN_DURATION:
                self.mode = AppMode.WORKOUT
                self.battle_start = time.time()
                self.p1.mode = AppMode.WORKOUT
                self.p1.workout_start = self.battle_start
                self.p1.reset_rep_machine()
                self.p2.mode = AppMode.WORKOUT
                self.p2.workout_start = self.battle_start
                self.p2.reset_rep_machine()
        elif self.mode == AppMode.WORKOUT:
            self.elapsed = now - self.battle_start
            if self.elapsed >= self.battle_duration:
                self.mode = AppMode.STOPPED
                self._determine_winner("TIME EXPIRED (60s)")
                self.p1.mode = AppMode.STOPPED
                self.p2.mode = AppMode.STOPPED

        # Process frame for this individual player (full camera frame!)
        engine = self.p1 if player == 1 else self.p2
        engine.mode = self.mode
        res = engine.process_frame(frame_bgr)

        # Calculate Tug-of-War Dominance:
        # Base: 50%. Each rep difference shifts the bar by 8%.
        diff = self.p1.good_reps - self.p2.good_reps
        dom_p1 = max(5.0, min(95.0, 50.0 + diff * 8.0))
        dom_p2 = 100.0 - dom_p1

        # Knockout trigger (+6 rep lead = 100% dominance)
        if self.mode == AppMode.WORKOUT:
            if diff >= 6:
                self.mode = AppMode.STOPPED
                self.winner = "P1"
                self.win_reason = f"DOMINANCE KNOCKOUT! (+{diff} REPS)"
                dom_p1 = 100.0
                dom_p2 = 0.0
            elif diff <= -6:
                self.mode = AppMode.STOPPED
                self.winner = "P2"
                self.win_reason = f"DOMINANCE KNOCKOUT! (+{-diff} REPS)"
                dom_p1 = 0.0
                dom_p2 = 100.0

        self.p1_dominance = round(dom_p1, 1)
        self.p2_dominance = round(dom_p2, 1)

        countdown_rem = 0.0
        timer_str = "01:00.0"
        if self.mode == AppMode.COUNTDOWN:
            countdown_rem = max(0.0, COUNTDOWN_DURATION - (now - self.countdown_start))
            timer_str = f"PREP: {countdown_rem:04.1f}s"
        elif self.mode == AppMode.WORKOUT:
            rem = max(0.0, self.battle_duration - self.elapsed)
            m, s = divmod(int(rem), 60)
            tenths = int((rem - int(rem)) * 10)
            timer_str = f"{m:02d}:{s:02d}.{tenths}"
        elif self.mode == AppMode.STOPPED:
            timer_str = "MATCH OVER"

        leader = "TIE"
        if self.p1.good_reps > self.p2.good_reps:
            leader = "P1"
        elif self.p2.good_reps > self.p1.good_reps:
            leader = "P2"

        return {
            "mode": self.mode.value,
            "player": player,
            "p1_ready": self.p1_ready,
            "p2_ready": self.p2_ready,
            "timer_str": timer_str,
            "countdown_rem": round(countdown_rem, 1),
            "p1_dominance": self.p1_dominance,
            "p2_dominance": self.p2_dominance,
            "leader": leader,
            "score_diff": diff,
            "winner": self.winner,
            "win_reason": self.win_reason,
            "my_result": res,
            "p1_reps": self.p1.good_reps,
            "p2_reps": self.p2.good_reps,
            "p1_bad": self.p1.bad_reps,
            "p2_bad": self.p2.bad_reps,
        }

    def process_split_frame(self, frame_bgr):
        # Fallback for split screen
        now = time.time()
        h, w = frame_bgr.shape[:2]
        mid = w // 2
        crop1 = frame_bgr[:, :mid]
        crop2 = frame_bgr[:, mid:]
        res1 = self.process_single_frame(1, crop1)
        res2 = self.process_single_frame(2, crop2)
        return res1


# ============================== FASTAPI APPLICATION ==============================
app = FastAPI(title="REX.FITNESS STUDIO - Pushup Coach")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

engine = WorkoutEngine()


class FramePayload(BaseModel):
    image: str


@app.post("/api/start")
def api_start():
    engine.start()
    return {"status": "ok", "mode": engine.mode.value}


@app.post("/api/stop")
def api_stop():
    engine.stop()
    return {
        "status": "ok",
        "mode": engine.mode.value,
        "duration": engine.workout_duration,
        "good_reps": engine.good_reps,
        "bad_reps": engine.bad_reps
    }


@app.post("/api/process_frame")
def api_process_frame(payload: FramePayload):
    try:
        header, encoded = payload.image.split(",", 1) if "," in payload.image else ("", payload.image)
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Failed to decode frame"}
        return engine.process_frame(frame)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/status")
def api_status():
    return {
        "mode": engine.mode.value,
        "state": engine.state,
        "good_reps": engine.good_reps,
        "bad_reps": engine.bad_reps,
        "elbow_angle": engine.elbow_angle,
        "body_angle": engine.body_angle,
    }


# ============================== 2-PLAYER BATTLE API ENDPOINTS ==============================
battle_engine = BattleEngine()


class PlayerFramePayload(BaseModel):
    player: int  # 1 or 2
    image: str


class ReadyPayload(BaseModel):
    player: int
    ready: bool = True


@app.post("/api/battle/ready")
def api_battle_ready(payload: ReadyPayload):
    battle_engine.set_ready(payload.player, payload.ready)
    return {
        "status": "ok",
        "p1_ready": battle_engine.p1_ready,
        "p2_ready": battle_engine.p2_ready,
        "mode": battle_engine.mode.value
    }


@app.post("/api/battle/frame_player")
def api_battle_frame_player(payload: PlayerFramePayload):
    try:
        header, encoded = payload.image.split(",", 1) if "," in payload.image else ("", payload.image)
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Failed to decode frame"}
        return battle_engine.process_single_frame(payload.player, frame)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/battle/start")
def api_battle_start():
    battle_engine.start()
    return {"status": "ok", "mode": battle_engine.mode.value}


@app.post("/api/battle/stop")
def api_battle_stop():
    battle_engine.stop()
    return {
        "status": "ok",
        "mode": battle_engine.mode.value,
        "winner": battle_engine.winner,
        "win_reason": battle_engine.win_reason,
        "p1_reps": battle_engine.p1.good_reps,
        "p2_reps": battle_engine.p2.good_reps,
    }


@app.post("/api/battle/reset")
def api_battle_reset():
    battle_engine.reset()
    return {"status": "ok", "mode": battle_engine.mode.value}


@app.post("/api/battle/process_frame")
def api_battle_process_frame(payload: FramePayload):
    try:
        header, encoded = payload.image.split(",", 1) if "," in payload.image else ("", payload.image)
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Failed to decode frame"}
        return battle_engine.process_split_frame(frame)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/battle/status")
def api_battle_status():
    return {
        "mode": battle_engine.mode.value,
        "winner": battle_engine.winner,
        "p1_ready": battle_engine.p1_ready,
        "p2_ready": battle_engine.p2_ready,
        "p1_reps": battle_engine.p1.good_reps,
        "p2_reps": battle_engine.p2.good_reps,
        "p1_dominance": battle_engine.p1_dominance,
        "p2_dominance": battle_engine.p2_dominance,
    }


# ========================= COMMON STYLES =========================
COMMON_HEAD = """
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Plus+Jakarta+Sans:wght@500;600;700;800;900&family=Space+Mono:wght@700&display=swap" rel="stylesheet">
  <style>
    :root {
      --brand-yellow: #FACC15;
      --brand-black: #0E0E0E;
      --brand-paper: #FAF9F5;
      --brand-border: #111111;
    }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: var(--brand-paper);
      color: var(--brand-black);
    }
    .font-bebas { font-family: 'Bebas Neue', cursive; }
    .font-mono { font-family: 'Space Mono', monospace; }

    .hard-shadow { box-shadow: 4px 4px 0px var(--brand-border); }
    .hard-shadow-lg { box-shadow: 7px 7px 0px var(--brand-border); }
    .hard-shadow-sm { box-shadow: 3px 3px 0px var(--brand-border); }
    .hard-button:hover {
      transform: translate(-2px, -2px);
      box-shadow: 6px 6px 0px var(--brand-border);
    }
    .hard-button:active {
      transform: translate(2px, 2px);
      box-shadow: 2px 2px 0px var(--brand-border);
    }
    .bg-dots {
      background-image: radial-gradient(#111111 1.2px, transparent 1.2px);
      background-size: 16px 16px;
    }
    .comic-card {
      background: #FFFFFF;
      border: 3px solid var(--brand-border);
      border-radius: 16px;
      box-shadow: 6px 6px 0px var(--brand-border);
    }
  </style>
"""


# ===================================================================
# 1. LANDING PAGE ROUTE (GET /)
# ===================================================================
@app.get("/", response_class=HTMLResponse)
def page_landing():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <title>REX.FITNESS STUDIO — Strict AI Pushup Coach</title>
  {COMMON_HEAD}
</head>
<body class="min-h-screen flex flex-col justify-between">

  <!-- Top Announcement Bar -->
  <div class="bg-black text-yellow-400 text-xs font-bold py-2 px-4 text-center tracking-wider border-b-2 border-black flex items-center justify-center gap-3">
    <span class="inline-block w-2 h-2 rounded-full bg-yellow-400 animate-ping"></span>
    <span>REX.FITNESS STUDIO &bull; AI COMPUTER VISION STRICT FORM ENGINE ACTIVE &bull; NO CHEAT REPS</span>
  </div>

  <!-- Navigation -->
  <nav class="sticky top-0 z-40 bg-[#FAF9F5]/95 backdrop-blur-md border-b-3 border-black px-4 md:px-8 py-4">
    <div class="max-w-7xl mx-auto flex items-center justify-between">
      
      <!-- Logo Branding -->
      <a href="/" class="flex items-center gap-2.5 group">
        <div class="w-10 h-10 bg-yellow-400 border-2 border-black rounded-lg flex items-center justify-center hard-shadow-sm font-bebas text-2xl text-black">
          R
        </div>
        <div class="flex flex-col">
          <span class="font-bebas text-2xl tracking-wider leading-none text-black group-hover:text-yellow-600 transition">REX.FITNESS STUDIO</span>
          <span class="text-[10px] font-extrabold uppercase tracking-widest text-slate-500">One-More Pushup AI</span>
        </div>
      </a>

      <!-- Desktop Nav Links -->
      <div class="hidden md:flex items-center gap-8 font-bold text-sm tracking-wide">
        <a href="/studio" class="hover:text-yellow-600 transition flex items-center gap-1.5 text-black">
          <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
          Live Studio
        </a>
        <a href="/battle" class="hover:text-yellow-600 transition flex items-center gap-1.5 text-black">
          <i class="fa-solid fa-bolt text-yellow-500"></i>
          1v1 Battle
        </a>
        <a href="/camera" class="hover:text-yellow-600 transition">Camera Setup</a>
      </div>

      <!-- Header CTA -->
      <div class="flex items-center gap-3">
        <a href="/battle" class="px-5 py-2.5 bg-yellow-400 text-black font-extrabold text-sm uppercase tracking-wider rounded-xl border-2 border-black hard-shadow-sm hard-button transition flex items-center gap-2">
          <i class="fa-solid fa-bolt"></i>
          <span>Launch 1v1 Battle</span>
        </a>
      </div>
    </div>
  </nav>

  <!-- ======================== HERO SECTION ======================== -->
  <section class="relative py-12 md:py-20 px-4 md:px-8 overflow-hidden bg-white flex-1 flex items-center">
    <div class="absolute right-0 top-0 w-96 h-96 bg-dots opacity-10 pointer-events-none"></div>
    <div class="absolute left-[-50px] bottom-[-50px] w-80 h-80 rounded-full bg-yellow-400/20 blur-3xl pointer-events-none"></div>

    <div class="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-10 items-center w-full">
      
      <!-- Hero Left: Copy & CTAs (7 Cols) -->
      <div class="lg:col-span-7 flex flex-col items-start">
        
        <!-- Category Badge -->
        <div class="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-yellow-400 border-2 border-black text-xs font-black uppercase tracking-wider hard-shadow-sm mb-6">
          <i class="fa-solid fa-arrows-left-right"></i>
          <span>Computer Vision Pushup Counter</span>
        </div>

        <!-- Big Bold Title with Yellow Accent -->
        <h1 class="font-bebas text-6xl md:text-8xl lg:text-9xl tracking-tight leading-[0.88] text-black mb-8">
          PUSH <span class="text-yellow-400 inline-block drop-shadow-[4px_4px_0px_#000000]">UP.</span><br>
          <span>STRICT FORM ONLY.</span>
        </h1>

        <!-- CTA Buttons -->
        <div class="flex flex-wrap items-center gap-4 w-full sm:w-auto">
          <a href="/studio" class="w-full sm:w-auto px-7 py-4 bg-yellow-400 text-black font-extrabold text-base uppercase tracking-wider rounded-2xl border-3 border-black hard-shadow-lg hard-button transition flex items-center justify-center gap-2">
            <i class="fa-solid fa-play"></i>
            <span>Solo Studio</span>
          </a>
          <a href="/battle" class="w-full sm:w-auto px-7 py-4 bg-black text-yellow-400 hover:bg-slate-900 font-extrabold text-base uppercase tracking-wider rounded-2xl border-3 border-black hard-shadow-lg hard-button transition flex items-center justify-center gap-2">
            <i class="fa-solid fa-bolt text-yellow-400"></i>
            <span>1v1 Pushup Battle</span>
          </a>
          <a href="/camera" class="w-full sm:w-auto px-6 py-4 bg-white text-black font-extrabold text-base uppercase tracking-wider rounded-2xl border-3 border-black hard-shadow hard-button transition flex items-center justify-center gap-2 hover:bg-slate-50">
            <i class="fa-solid fa-video"></i>
            <span>Camera Setup</span>
          </a>
        </div>

      </div>

      <!-- Hero Right: Brand Illustration (5 Cols) -->
      <div class="lg:col-span-5 flex justify-center">
        <div class="relative w-full max-w-lg">
          <div class="absolute inset-0 bg-yellow-400 border-3 border-black rounded-3xl translate-x-3 translate-y-3 md:translate-x-4 md:translate-y-4"></div>
          
          <div class="relative bg-white border-3 border-black rounded-3xl p-6 md:p-8 flex flex-col items-center overflow-hidden">
            <div class="w-full aspect-[4/3] rounded-2xl overflow-hidden border-2 border-black bg-white flex items-center justify-center p-2 mb-4">
              <img src="/static/hero-pushup.jpg" alt="REX.FITNESS STUDIO Pushup Illustration" class="w-full h-full object-contain">
            </div>

            <div class="w-full flex items-center justify-between pt-2">
              <span class="font-bebas text-xl tracking-wider text-black">NO HALF-REPS &bull; STRICT REPS ONLY</span>
              <span class="text-xs font-black uppercase px-2.5 py-1 bg-black text-yellow-400 rounded-md">AI Certified</span>
            </div>
          </div>
        </div>
      </div>

    </div>
  </section>

  <!-- Footer -->
  <footer class="bg-black text-white py-8 px-4 md:px-8 border-t-3 border-black">
    <div class="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 bg-yellow-400 border-2 border-white rounded-lg flex items-center justify-center font-bebas text-lg text-black">
          R
        </div>
        <span class="font-bebas text-xl tracking-wider text-white">REX.FITNESS STUDIO</span>
      </div>
      <div class="text-xs font-bold text-slate-400">
        <span>No cheat reps. Only strict form. &bull; One-More Pushup AI</span>
      </div>
    </div>
  </footer>

</body>
</html>
"""


# ===================================================================
# 2. DEDICATED LIVE STUDIO PAGE ROUTE (GET /studio)
# ===================================================================
@app.get("/studio", response_class=HTMLResponse)
def page_studio():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <title>Live AI Studio — REX.FITNESS STUDIO</title>
  {COMMON_HEAD}
</head>
<body class="min-h-screen flex flex-col bg-[#FAF9F5]">

  <!-- Studio Top Navigation -->
  <header class="bg-white border-b-3 border-black px-4 md:px-8 py-3.5 sticky top-0 z-40">
    <div class="max-w-7xl mx-auto flex items-center justify-between">
      
      <!-- Back to Home Link -->
      <a href="/" class="flex items-center gap-2 text-black hover:text-yellow-600 font-extrabold text-sm uppercase tracking-wider transition group">
        <div class="w-8 h-8 bg-yellow-400 border-2 border-black rounded-lg flex items-center justify-center hard-shadow-sm group-hover:-translate-x-0.5 transition">
          <i class="fa-solid fa-arrow-left text-black text-xs"></i>
        </div>
        <span>Back to Home</span>
      </a>

      <!-- Studio Center Brand -->
      <div class="flex items-center gap-2">
        <div class="w-8 h-8 bg-black text-yellow-400 border-2 border-black rounded-lg flex items-center justify-center font-bebas text-lg">
          R
        </div>
        <span class="font-bebas text-2xl tracking-wider text-black hidden sm:inline">AI WORKOUT STUDIO</span>
      </div>

      <!-- Studio Mode Tag & Guide Link -->
      <div class="flex items-center gap-3">
        <a href="/battle" class="inline-flex items-center gap-1.5 text-xs font-extrabold uppercase px-3 py-1.5 bg-yellow-400 border-2 border-black rounded-lg hover:bg-yellow-300 transition hard-shadow-sm">
          <i class="fa-solid fa-bolt"></i>
          <span>1v1 Battle</span>
        </a>
        <a href="/camera" class="hidden md:inline-flex items-center gap-1.5 text-xs font-extrabold uppercase px-3 py-1.5 bg-[#FAF9F5] border-2 border-black rounded-lg hover:bg-yellow-400 transition">
          <i class="fa-solid fa-video"></i>
          <span>Camera Guide</span>
        </a>
        <span id="badgeMode" class="px-3 py-1.5 rounded-lg text-xs font-black uppercase bg-black text-yellow-400 border-2 border-black">
          STANDBY
        </span>
      </div>
    </div>
  </header>

  <!-- Studio Main Content -->
  <main class="max-w-[1700px] w-full mx-auto p-3 sm:p-5 lg:p-6 flex-1 flex flex-col justify-center">

    <!-- Studio Interactive Container (12 Cols) -->
    <div class="comic-card p-4 sm:p-5 lg:p-6 bg-white hard-shadow-lg">
      
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">

        <!-- Massive Video Viewport (9 Cols) -->
        <div class="lg:col-span-9 flex flex-col">
          <div class="relative w-full h-[68vh] min-h-[520px] max-h-[840px] bg-slate-950 rounded-2xl overflow-hidden border-3 border-black hard-shadow flex items-center justify-center">

            <!-- Live Camera & Canvas -->
            <video id="webcam" autoplay playsinline muted class="hidden"></video>
            <canvas id="outputCanvas" class="w-full h-full object-contain bg-slate-950"></canvas>

            <!-- Top Floating HUD Pill -->
            <div id="hudPill" class="absolute top-4 left-1/2 -translate-x-1/2 px-5 py-2 rounded-xl bg-black/95 text-white border-2 border-yellow-400 text-xs md:text-sm font-extrabold tracking-wide hard-shadow-sm flex items-center gap-2 z-10">
              <span class="w-2.5 h-2.5 rounded-full bg-yellow-400 animate-ping"></span>
              <span id="hudPillText">Position camera at side view & click START</span>
            </div>

            <!-- Toast Feedback Banner (Good Rep / Bad Rep) -->
            <div id="toastFeedback" class="hidden absolute top-16 left-1/2 -translate-x-1/2 px-6 py-2.5 rounded-xl text-base md:text-lg font-black tracking-wide border-3 border-black hard-shadow-lg z-20 transition-all duration-300">
            </div>

            <!-- 10-Second Preparation Countdown Modal -->
            <div id="countdownOverlay" class="hidden absolute inset-0 bg-black/85 backdrop-blur-sm flex flex-col items-center justify-center p-6 text-center z-30">
              <div class="p-6 md:p-8 rounded-3xl bg-yellow-400 border-4 border-black hard-shadow-lg max-w-sm w-full flex flex-col items-center">
                <span class="font-bebas text-2xl text-black tracking-widest uppercase">PREPARATION TIME</span>
                <div class="my-4 w-32 h-32 rounded-full bg-white border-4 border-black flex items-center justify-center hard-shadow-sm">
                  <span id="countdownNumber" class="font-bebas text-7xl text-black leading-none">10</span>
                </div>
                <h3 class="text-lg font-black text-black mb-1">Get Into Plank Position!</h3>
                <p class="text-xs font-bold text-black/80 mb-4">Step back into side view. Workout timer begins when countdown reaches zero.</p>
                <div class="w-full h-3 bg-white border-2 border-black rounded-full overflow-hidden">
                  <div id="countdownBar" class="h-full bg-black transition-all duration-200" style="width: 100%"></div>
                </div>
              </div>
            </div>

            <!-- Workout Summary Scorecard Modal -->
            <div id="summaryOverlay" class="hidden absolute inset-0 bg-black/90 backdrop-blur-md flex flex-col items-center justify-center p-6 z-40">
              <div class="bg-white border-4 border-black p-6 md:p-8 rounded-3xl max-w-md w-full text-center hard-shadow-lg">
                <div class="w-16 h-16 mx-auto rounded-2xl bg-yellow-400 border-3 border-black flex items-center justify-center text-3xl mb-3 hard-shadow-sm">
                  <i class="fa-solid fa-trophy text-black"></i>
                </div>
                <h3 class="font-bebas text-4xl text-black leading-none mb-1">WORKOUT COMPLETE!</h3>
                <p class="text-xs font-bold uppercase tracking-wider text-slate-500 mb-6">Official Form Scorecard</p>

                <div class="grid grid-cols-2 gap-3 mb-6">
                  <div class="p-3 bg-yellow-50 border-2 border-black rounded-xl hard-shadow-sm">
                    <div class="text-[11px] font-extrabold uppercase text-slate-600">Strict Reps</div>
                    <div id="sumGood" class="font-bebas text-4xl text-black leading-none mt-1">0</div>
                  </div>
                  <div class="p-3 bg-red-50 border-2 border-black rounded-xl hard-shadow-sm">
                    <div class="text-[11px] font-extrabold uppercase text-slate-600">No Point</div>
                    <div id="sumBad" class="font-bebas text-4xl text-red-600 leading-none mt-1">0</div>
                  </div>
                  <div class="p-3 bg-[#FAF9F5] border-2 border-black rounded-xl hard-shadow-sm">
                    <div class="text-[11px] font-extrabold uppercase text-slate-600">Session Time</div>
                    <div id="sumDuration" class="font-mono text-xl font-bold text-black mt-1">00:00</div>
                  </div>
                  <div class="p-3 bg-[#FAF9F5] border-2 border-black rounded-xl hard-shadow-sm">
                    <div class="text-[11px] font-extrabold uppercase text-slate-600">Form Accuracy</div>
                    <div id="sumAccuracy" class="font-bebas text-3xl text-emerald-600 leading-none mt-1">100%</div>
                  </div>
                </div>

                <button onclick="triggerStart()" class="w-full py-4 bg-yellow-400 text-black font-extrabold text-base uppercase tracking-wider rounded-2xl border-3 border-black hard-shadow hard-button transition">
                  Start New Workout
                </button>
              </div>
            </div>

          </div>

          <!-- Viewport Bottom Hint -->
          <div class="mt-3 flex items-center justify-between text-xs font-bold text-slate-500 px-1">
            <span><i class="fa-solid fa-video mr-1"></i> Full side view camera &bull; Live posture angles on skeleton</span>
            <span><i class="fa-solid fa-keyboard mr-1"></i> Press [S] to Start &bull; [Space] to Stop</span>
          </div>
        </div>

        <!-- Studio Dashboard Controls (3 Cols) -->
        <div class="lg:col-span-3 flex flex-col justify-between gap-4">
          
          <div class="flex flex-col gap-4">
            <!-- Start & Stop Action Buttons -->
            <div class="grid grid-cols-2 gap-3">
              <button id="btnStart" onclick="triggerStart()" class="py-4 bg-yellow-400 text-black font-extrabold text-base md:text-lg uppercase tracking-wider rounded-2xl border-3 border-black hard-shadow hard-button flex items-center justify-center gap-2 cursor-pointer transition">
                <i class="fa-solid fa-play"></i>
                <span>START</span>
              </button>
              <button id="btnStop" onclick="triggerStop()" disabled class="py-4 bg-slate-200 text-slate-400 font-extrabold text-base md:text-lg uppercase tracking-wider rounded-2xl border-3 border-slate-300 cursor-not-allowed flex items-center justify-center gap-2 transition">
                <i class="fa-solid fa-stop"></i>
                <span>STOP</span>
              </button>
            </div>

            <!-- Session Timer Card -->
            <div class="p-4 bg-black text-white border-3 border-black rounded-2xl hard-shadow">
              <div class="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-yellow-400 mb-1">
                <span>Workout Session Timer</span>
                <i class="fa-solid fa-stopwatch"></i>
              </div>
              <div id="timerDisplay" class="font-mono text-4xl md:text-5xl font-extrabold tracking-tight text-white">00:00.0</div>
              <p id="timerDesc" class="text-[11px] font-bold text-slate-400 mt-1">Starts automatically 10s after clicking Start</p>
            </div>

            <!-- Reps Counter Cards -->
            <div class="flex flex-col gap-3">
              <!-- Strict Reps -->
              <div class="p-5 bg-yellow-400 border-3 border-black rounded-2xl hard-shadow">
                <div class="flex items-center justify-between">
                  <span class="text-xs font-black uppercase tracking-wider text-black">Strict Reps</span>
                  <span class="text-[10px] font-extrabold px-2 py-0.5 bg-black text-yellow-400 rounded-md uppercase">&le; 90&deg; DEPTH</span>
                </div>
                <div id="countGood" class="font-bebas text-7xl md:text-8xl text-black leading-none mt-2">0</div>
                <div class="text-[11px] font-extrabold text-black/80 mt-1">Full Lockout & Straight Spine</div>
              </div>

              <!-- No Point Reps -->
              <div class="p-4 bg-white border-3 border-black rounded-2xl hard-shadow">
                <div class="flex items-center justify-between">
                  <span class="text-xs font-black uppercase tracking-wider text-red-600">No Point</span>
                  <span class="text-[10px] font-bold text-slate-500 uppercase">REJECTED</span>
                </div>
                <div id="countBad" class="font-bebas text-6xl text-red-600 leading-none mt-1">0</div>
                <div class="text-[11px] font-bold text-slate-500 mt-0.5">Shallow depth or sagging spine</div>
              </div>
            </div>
          </div>

          <!-- Studio Quick Calibration Note -->
          <div class="p-3 bg-[#FAF9F5] border-2 border-black rounded-xl text-xs font-bold text-slate-600">
            <div class="flex items-center gap-1.5 text-black font-extrabold mb-1">
              <i class="fa-solid fa-circle-info text-yellow-500"></i>
              <span>POSTURE GUIDE</span>
            </div>
            <p class="text-[11px] text-slate-600 leading-relaxed">
              Step 2&ndash;3 meters back so full body is visible. Joint angles & depth guides appear directly on your posture in real time.
            </p>
          </div>

        </div>

      </div>

    </div>

  </main>

      </div>

    </div>

  </main>

  <!-- Client Script for Studio -->
  <script>
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('outputCanvas');
    const ctx = canvas.getContext('2d');

    let isProcessing = false;
    let lastFrameTime = performance.now();
    let frameCount = 0;
    let lastAudioAnnounce = -1;

    navigator.mediaDevices.getUserMedia({{ video: {{ width: 640, height: 480 }} }})
      .then(stream => {{
        video.srcObject = stream;
        video.play();
        requestAnimationFrame(processLoop);
      }})
      .catch(err => {{
        console.warn('Webcam access error:', err);
        document.getElementById('hudPillText').innerText = "Camera access needed: please allow browser permissions.";
      }});

    async function processLoop() {{
      if (video.readyState === 4 && !isProcessing) {{
        isProcessing = true;
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.55);

        try {{
          const res = await fetch('/api/process_frame', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ image: dataUrl }})
          }});
          const data = await res.json();
          if (!data.error) {{
            updateDashboard(data);
            drawSkeleton(data);
          }}
        }} catch (e) {{
          console.error(e);
        }} finally {{
          isProcessing = false;
        }}

        frameCount++;
        const now = performance.now();
        if (now - lastFrameTime >= 1000) {{
          const fpsEl = document.getElementById('badgeFPS');
          if (fpsEl) fpsEl.innerText = `${{frameCount}} FPS`;
          frameCount = 0;
          lastFrameTime = now;
        }}
      }}
      requestAnimationFrame(processLoop);
    }}

    function drawSkeleton(data) {{
      if (!data.landmarks || data.landmarks.length < 5) return;
      const pts = data.landmarks.map(p => ({{ x: p.x * canvas.width, y: p.y * canvas.height }}));
      const [sh, el, wr, hip, ank] = pts;

      const isGood = data.is_good_form;
      const color = isGood ? '#FACC15' : '#EF4444';

      // 1. High contrast dark border outline (visible across any room background)
      ctx.lineWidth = 10;
      ctx.strokeStyle = '#000000';
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      // Arm bones
      ctx.moveTo(sh.x, sh.y); ctx.lineTo(el.x, el.y); ctx.lineTo(wr.x, wr.y);
      // Torso & leg bones
      ctx.moveTo(sh.x, sh.y); ctx.lineTo(hip.x, hip.y); ctx.lineTo(ank.x, ank.y);
      ctx.stroke();

      // 2. High visibility colored core
      ctx.lineWidth = 6;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(sh.x, sh.y); ctx.lineTo(el.x, el.y); ctx.lineTo(wr.x, wr.y);
      ctx.moveTo(sh.x, sh.y); ctx.lineTo(hip.x, hip.y); ctx.lineTo(ank.x, ank.y);
      ctx.stroke();

      // 3. Highlighted Joints
      pts.forEach(p => {{
        ctx.fillStyle = '#000000';
        ctx.beginPath(); ctx.arc(p.x, p.y, 9, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI * 2); ctx.fill();
      }});

      // 4. Live Posture Angle Badges directly on Joints
      function drawBadge(x, y, text, subtext, bgColor, textColor) {{
        ctx.save();
        ctx.font = 'bold 15px monospace';
        const tWidth = ctx.measureText(text).width;
        const bWidth = Math.max(tWidth + 28, 76);
        const bHeight = subtext ? 36 : 24;
        const bx = Math.min(Math.max(x - bWidth / 2, 12), canvas.width - bWidth - 12);
        const by = Math.min(Math.max(y - 42, 12), canvas.height - bHeight - 12);

        // Shadow & Border
        ctx.fillStyle = '#000000';
        ctx.fillRect(bx + 2, by + 2, bWidth, bHeight);
        ctx.fillStyle = bgColor;
        ctx.fillRect(bx, by, bWidth, bHeight);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#000000';
        ctx.strokeRect(bx, by, bWidth, bHeight);

        ctx.fillStyle = textColor;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (subtext) {{
          ctx.font = '900 14px monospace';
          ctx.fillText(text, bx + bWidth / 2, by + 13);
          ctx.font = 'bold 9px sans-serif';
          ctx.fillText(subtext, bx + bWidth / 2, by + 26);
        }} else {{
          ctx.font = '900 14px monospace';
          ctx.fillText(text, bx + bWidth / 2, by + bHeight / 2);
        }}
        ctx.restore();
      }}

      // Live Elbow Angle Badge
      if (data.elbow_angle !== undefined) {{
        const elDeg = Math.round(data.elbow_angle);
        const elBg = elDeg <= 90 ? '#22C55E' : (elDeg <= 120 ? '#FACC15' : '#FFFFFF');
        const elSub = elDeg <= 90 ? 'DEPTH OK' : (elDeg > 150 ? 'LOCKED' : 'LOWER');
        drawBadge(el.x, el.y, `${{elDeg}}°`, elSub, elBg, '#000000');
      }}

      // Live Spine Alignment Badge
      if (data.body_angle !== undefined) {{
        const bdDeg = Math.round(data.body_angle);
        const bdBg = bdDeg >= 160 ? '#22C55E' : '#EF4444';
        const bdSub = bdDeg >= 160 ? 'SPINE OK' : 'SAG / PIKE';
        const txtCol = bdDeg >= 160 ? '#000000' : '#FFFFFF';
        drawBadge(hip.x, hip.y, `${{bdDeg}}°`, bdSub, bdBg, txtCol);
      }}
    }}

    function speak(text) {{
      if ('speechSynthesis' in window) {{
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.25;
        window.speechSynthesis.speak(utter);
      }}
    }}

    function updateDashboard(data) {{
      const textMode = document.getElementById('textModeStatus');
      if (textMode) textMode.innerText = data.mode;
      const badgeMode = document.getElementById('badgeMode');
      if (badgeMode) badgeMode.innerText = data.mode;

      const btnStart = document.getElementById('btnStart');
      const btnStop = document.getElementById('btnStop');

      if (data.mode === 'IDLE' || data.mode === 'STOPPED') {{
        btnStart.disabled = false;
        btnStart.className = "py-4 bg-yellow-400 hover:bg-yellow-300 text-black font-extrabold text-base md:text-lg uppercase tracking-wider rounded-2xl border-3 border-black hard-shadow hard-button flex items-center justify-center gap-2 cursor-pointer transition";
        btnStop.disabled = true;
        btnStop.className = "py-4 bg-slate-200 text-slate-400 font-extrabold text-base md:text-lg uppercase tracking-wider rounded-2xl border-3 border-slate-300 cursor-not-allowed flex items-center justify-center gap-2 transition";
      }} else {{
        btnStart.disabled = true;
        btnStart.className = "py-4 bg-slate-200 text-slate-400 font-extrabold text-base md:text-lg uppercase tracking-wider rounded-2xl border-3 border-slate-300 cursor-not-allowed flex items-center justify-center gap-2 transition";
        btnStop.disabled = false;
        btnStop.className = "py-4 bg-red-600 hover:bg-red-500 text-white font-extrabold text-base md:text-lg uppercase tracking-wider rounded-2xl border-3 border-black hard-shadow hard-button flex items-center justify-center gap-2 cursor-pointer transition";
      }}

      const timerEl = document.getElementById('timerDisplay');
      if (timerEl) timerEl.innerText = data.timer_str;

      const cdOverlay = document.getElementById('countdownOverlay');
      if (data.mode === 'COUNTDOWN') {{
        cdOverlay.classList.remove('hidden');
        const remInt = Math.ceil(data.countdown_rem);
        document.getElementById('countdownNumber').innerText = remInt;
        document.getElementById('countdownBar').style.width = `${{((10.0 - data.countdown_rem) / 10.0) * 100}}%`;
        if (remInt <= 3 && remInt > 0 && remInt !== lastAudioAnnounce) {{
          lastAudioAnnounce = remInt;
          speak(String(remInt));
        }}
      }} else {{
        if (!cdOverlay.classList.contains('hidden') && lastAudioAnnounce !== -1) {{
          speak("Go!");
          lastAudioAnnounce = -1;
        }}
        cdOverlay.classList.add('hidden');
      }}

      const countGoodEl = document.getElementById('countGood');
      if (countGoodEl) countGoodEl.innerText = data.good_reps;
      const countBadEl = document.getElementById('countBad');
      if (countBadEl) countBadEl.innerText = data.bad_reps;

      let hudMsg = data.status_msg;
      if (data.mode === 'WORKOUT') {{
        if (data.state === 'WAIT') hudMsg = "Hold still in plank for 1s...";
        else if (data.state === 'UP') hudMsg = "Ready — lower your chest";
        else if (data.state === 'LOWERING') hudMsg = "Going down... (reach 90°)";
        else if (data.state === 'DOWN') hudMsg = "Full depth! Push up!";
      }}
      const hudPillText = document.getElementById('hudPillText');
      if (hudPillText) hudPillText.innerText = hudMsg;

      const toast = document.getElementById('toastFeedback');
      if (data.feedback) {{
        toast.innerText = data.feedback;
        toast.classList.remove('hidden');
        if (data.feedback.includes('GOOD')) {{
          toast.className = "absolute top-16 left-1/2 -translate-x-1/2 px-6 py-2.5 rounded-xl text-base md:text-lg font-black tracking-wide border-3 border-black hard-shadow-lg z-20 bg-yellow-400 text-black";
        }} else {{
          toast.className = "absolute top-16 left-1/2 -translate-x-1/2 px-6 py-2.5 rounded-xl text-base md:text-lg font-black tracking-wide border-3 border-black hard-shadow-lg z-20 bg-red-600 text-white";
        }}
      }} else {{
        toast.classList.add('hidden');
      }}

      const sumModal = document.getElementById('summaryOverlay');
      if (data.mode === 'STOPPED') {{
        document.getElementById('sumGood').innerText = data.good_reps;
        document.getElementById('sumBad').innerText = data.bad_reps;
        document.getElementById('sumDuration').innerText = data.timer_str;
        document.getElementById('sumAccuracy').innerText = `${{data.accuracy}}%`;
        sumModal.classList.remove('hidden');
      }} else {{
        sumModal.classList.add('hidden');
      }}
    }}

    async function triggerStart() {{
      document.getElementById('summaryOverlay').classList.add('hidden');
      await fetch('/api/start', {{ method: 'POST' }});
      speak("10 second prep countdown started");
    }}

    async function triggerStop() {{
      await fetch('/api/stop', {{ method: 'POST' }});
      speak("Workout stopped");
    }}

    window.addEventListener('keydown', (e) => {{
      if (e.key === 's' || e.key === 'S') triggerStart();
      if (e.key === ' ' || e.key === 'x' || e.key === 'X') triggerStop();
    }});
  </script>
</body>
</html>
"""


# ===================================================================
# 3. DEDICATED CAMERA SETUP PAGE ROUTE (GET /camera)
# ===================================================================
@app.get("/camera", response_class=HTMLResponse)
def page_camera():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <title>Camera Setup Guide — REX.FITNESS STUDIO</title>
  {COMMON_HEAD}
</head>
<body class="min-h-screen flex flex-col bg-[#FAF9F5]">

  <!-- Top Navigation -->
  <header class="bg-white border-b-3 border-black px-4 md:px-8 py-3.5 sticky top-0 z-40">
    <div class="max-w-7xl mx-auto flex items-center justify-between">
      
      <!-- Back to Home Link -->
      <a href="/" class="flex items-center gap-2 text-black hover:text-yellow-600 font-extrabold text-sm uppercase tracking-wider transition group">
        <div class="w-8 h-8 bg-yellow-400 border-2 border-black rounded-lg flex items-center justify-center hard-shadow-sm group-hover:-translate-x-0.5 transition">
          <i class="fa-solid fa-arrow-left text-black text-xs"></i>
        </div>
        <span>Back to Home</span>
      </a>

      <!-- Brand -->
      <div class="flex items-center gap-2">
        <div class="w-8 h-8 bg-black text-yellow-400 border-2 border-black rounded-lg flex items-center justify-center font-bebas text-lg">
          R
        </div>
        <span class="font-bebas text-2xl tracking-wider text-black hidden sm:inline">CAMERA CALIBRATION</span>
      </div>

      <!-- Action Button -->
      <div class="flex items-center gap-2">
        <a href="/battle" class="px-4 py-2 bg-black text-yellow-400 font-extrabold text-xs uppercase tracking-wider rounded-xl border-2 border-black hard-shadow-sm hard-button transition flex items-center gap-1.5">
          <i class="fa-solid fa-bolt"></i>
          <span>1v1 Battle</span>
        </a>
        <a href="/studio" class="px-4 py-2 bg-yellow-400 text-black font-extrabold text-xs uppercase tracking-wider rounded-xl border-2 border-black hard-shadow-sm hard-button transition flex items-center gap-1.5">
          <i class="fa-solid fa-play"></i>
          <span>Launch Studio</span>
        </a>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="max-w-7xl w-full mx-auto p-4 md:p-8 flex-1 flex flex-col justify-center">

    <!-- Header Section -->
    <div class="text-center max-w-2xl mx-auto mb-10">
      <div class="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-yellow-400 border-2 border-black text-xs font-black uppercase tracking-wider hard-shadow-sm mb-3">
        <i class="fa-solid fa-video"></i>
        <span>AI Calibration Checklist</span>
      </div>
      <h1 class="font-bebas text-5xl md:text-7xl text-black leading-none mb-3">
        CAMERA POSITIONING GUIDE
      </h1>
      <p class="text-sm md:text-base font-semibold text-slate-700">
        To achieve 100% accurate pushup rep counting and biomechanical form scoring, follow these 3 camera positioning rules.
      </p>
    </div>

    <!-- 3 Core Rule Cards -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
      
      <!-- Rule 1 -->
      <div class="comic-card p-6 flex flex-col justify-between hard-shadow">
        <div>
          <div class="w-12 h-12 bg-yellow-400 border-2 border-black rounded-xl flex items-center justify-center font-bebas text-2xl text-black hard-shadow-sm mb-4">
            01
          </div>
          <h2 class="font-bebas text-3xl text-black mb-2">EXACT 90&deg; SIDE VIEW</h2>
          <p class="text-xs font-medium text-slate-700 leading-relaxed mb-4">
            Position your camera directly perpendicular to your body. Do NOT place it diagonally in front or behind you. The computer vision model needs a flat lateral view to measure the elbow and hip angles.
          </p>
        </div>
        <div class="p-3 bg-yellow-100/60 border-2 border-black rounded-xl text-[11px] font-bold text-black flex items-center gap-2">
          <i class="fa-solid fa-circle-check text-black"></i>
          <span>Perpendicular view required</span>
        </div>
      </div>

      <!-- Rule 2 -->
      <div class="comic-card p-6 flex flex-col justify-between hard-shadow bg-yellow-50">
        <div>
          <div class="w-12 h-12 bg-black text-yellow-400 border-2 border-black rounded-xl flex items-center justify-center font-bebas text-2xl hard-shadow-sm mb-4">
            02
          </div>
          <h2 class="font-bebas text-3xl text-black mb-2">LOW TO THE FLOOR</h2>
          <p class="text-xs font-medium text-slate-700 leading-relaxed mb-4">
            Set your phone or laptop roughly at chest or floor height (about 15 to 30 cm off the floor). A camera placed high looking downward distorts depth perception and can cause false rep counts.
          </p>
        </div>
        <div class="p-3 bg-white border-2 border-black rounded-xl text-[11px] font-bold text-black flex items-center gap-2">
          <i class="fa-solid fa-circle-check text-black"></i>
          <span>15–30 cm off ground</span>
        </div>
      </div>

      <!-- Rule 3 -->
      <div class="comic-card p-6 flex flex-col justify-between hard-shadow">
        <div>
          <div class="w-12 h-12 bg-yellow-400 border-2 border-black rounded-xl flex items-center justify-center font-bebas text-2xl text-black hard-shadow-sm mb-4">
            03
          </div>
          <h2 class="font-bebas text-3xl text-black mb-2">2 TO 3 METERS AWAY</h2>
          <p class="text-xs font-medium text-slate-700 leading-relaxed mb-4">
            Ensure your head, hands, hips, knees, and feet are all fully visible in frame when lying down in plank position. If any joint is clipped off-screen, the tracker will pause counting.
          </p>
        </div>
        <div class="p-3 bg-yellow-100/60 border-2 border-black rounded-xl text-[11px] font-bold text-black flex items-center gap-2">
          <i class="fa-solid fa-circle-check text-black"></i>
          <span>Head to toe framing</span>
        </div>
      </div>

    </div>

    <!-- Live Test Camera Card -->
    <div class="comic-card p-6 md:p-8 bg-white hard-shadow-lg max-w-4xl mx-auto w-full">
      <div class="flex flex-col md:flex-row items-center gap-6">
        
        <!-- Camera Preview Box -->
        <div class="w-full md:w-1/2 aspect-[4/3] bg-black rounded-2xl overflow-hidden border-3 border-black relative flex items-center justify-center">
          <video id="previewVideo" autoplay playsinline muted class="w-full h-full object-cover"></video>
          <div class="absolute bottom-3 left-3 px-3 py-1 bg-black/80 text-yellow-400 border border-yellow-400 rounded-lg text-xs font-black uppercase">
            Live Preview Check
          </div>
        </div>

        <!-- Checklist & Studio Button -->
        <div class="w-full md:w-1/2 flex flex-col justify-between">
          <div>
            <h2 class="font-bebas text-3xl text-black mb-2 leading-none">TEST YOUR FRAMING</h2>
            <p class="text-xs font-semibold text-slate-600 mb-4">Check that you are visible from head to toe in the preview before starting.</p>
            
            <div class="space-y-2 mb-6">
              <div class="flex items-center gap-2 text-xs font-bold text-slate-800">
                <i class="fa-solid fa-square-check text-yellow-500 text-sm"></i>
                <span>Can see hands and shoulders</span>
              </div>
              <div class="flex items-center gap-2 text-xs font-bold text-slate-800">
                <i class="fa-solid fa-square-check text-yellow-500 text-sm"></i>
                <span>Can see hips and knees</span>
              </div>
              <div class="flex items-center gap-2 text-xs font-bold text-slate-800">
                <i class="fa-solid fa-square-check text-yellow-500 text-sm"></i>
                <span>Can see toes and feet</span>
              </div>
            </div>
          </div>

          <a href="/studio" class="w-full py-4 bg-yellow-400 text-black font-extrabold text-sm uppercase tracking-wider rounded-2xl border-3 border-black hard-shadow hard-button transition flex items-center justify-center gap-2">
            <i class="fa-solid fa-bolt"></i>
            <span>Framing Confirmed — Go to Studio</span>
          </a>
        </div>

      </div>
    </div>

  </main>

  <!-- Footer -->
  <footer class="bg-black text-white py-6 px-4 md:px-8 border-t-3 border-black text-center text-xs font-bold text-slate-400">
    <span>REX.FITNESS STUDIO &bull; ONE-MORE PUSHUP AI</span>
  </footer>

  <script>
    const pVideo = document.getElementById('previewVideo');
    navigator.mediaDevices.getUserMedia({{ video: {{ width: 640, height: 480 }} }})
      .then(stream => {{ pVideo.srcObject = stream; pVideo.play(); }})
      .catch(err => console.warn(err));
  </script>
</body>
</html>
"""


# ===================================================================
# 4. 2-PLAYER LIVE 1v1 BATTLE PAGE ROUTE (GET /battle)
# ===================================================================
@app.get("/battle", response_class=HTMLResponse)
def page_battle():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <title>1v1 Live Pushup Battle — REX.FITNESS STUDIO</title>
  {COMMON_HEAD}
</head>
<body class="min-h-screen flex flex-col bg-[#FAF9F5]">

  <!-- Top Navigation -->
  <header class="bg-white border-b-3 border-black px-4 md:px-8 py-3 sticky top-0 z-40">
    <div class="max-w-[1700px] mx-auto flex items-center justify-between">
      
      <!-- Back to Home -->
      <a href="/" class="flex items-center gap-2 text-black hover:text-yellow-600 font-extrabold text-sm uppercase tracking-wider transition group">
        <div class="w-8 h-8 bg-yellow-400 border-2 border-black rounded-lg flex items-center justify-center hard-shadow-sm group-hover:-translate-x-0.5 transition">
          <i class="fa-solid fa-arrow-left text-black text-xs"></i>
        </div>
        <span>Home</span>
      </a>

      <!-- Center Title -->
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 bg-black text-yellow-400 border-2 border-black rounded-lg flex items-center justify-center font-bebas text-lg">
          VS
        </div>
        <span class="font-bebas text-2xl md:text-3xl tracking-wider text-black">1v1 PUSHUP BATTLE</span>
        <span id="badgeMode" class="hidden sm:inline-block px-3 py-1 rounded-lg text-xs font-black uppercase bg-black text-yellow-400 border-2 border-black">
          STANDBY
        </span>
      </div>

      <!-- Player Role Selector & Links -->
      <div class="flex items-center gap-3">
        
        <!-- Role Switcher -->
        <div class="flex items-center gap-1.5 bg-[#FAF9F5] p-1 border-2 border-black rounded-xl">
          <span class="text-[10px] font-black uppercase text-slate-500 px-1 hidden lg:inline">ROLE:</span>
          <button id="btnRole1" onclick="switchRole(1)" class="px-3 py-1 rounded-lg text-xs font-black uppercase bg-yellow-400 text-black border-2 border-black hard-shadow-sm transition">
            🟡 Player 1
          </button>
          <button id="btnRole2" onclick="switchRole(2)" class="px-3 py-1 rounded-lg text-xs font-black uppercase bg-white text-slate-700 border-2 border-black hover:bg-red-500 hover:text-white transition">
            🔴 Player 2
          </button>
        </div>

        <!-- Copy Opponent Link -->
        <button onclick="copyOpponentLink()" class="hidden md:inline-flex items-center gap-1.5 text-xs font-extrabold uppercase px-3 py-1.5 bg-white border-2 border-black rounded-lg hover:bg-slate-100 transition hard-shadow-sm">
          <i class="fa-solid fa-link text-xs"></i>
          <span>Copy P2 Link</span>
        </button>

        <a href="/studio" class="inline-flex items-center gap-1.5 text-xs font-extrabold uppercase px-3 py-1.5 bg-[#FAF9F5] border-2 border-black rounded-lg hover:bg-yellow-400 transition">
          <i class="fa-solid fa-user"></i>
          <span>Solo</span>
        </a>
      </div>
    </div>
  </header>

  <!-- Battle Main Arena -->
  <main class="max-w-[1700px] w-full mx-auto p-3 sm:p-5 flex-1 flex flex-col justify-center gap-4">

    <!-- ==================== TUG-OF-WAR DOMINANCE BAR ==================== -->
    <div class="comic-card p-3 md:p-4 bg-white hard-shadow-lg">
      <div class="flex flex-col md:flex-row items-center gap-4">

        <!-- P1 Player Badge (Left) -->
        <div class="flex items-center gap-3 min-w-[220px] justify-start w-full md:w-auto">
          <div class="w-12 h-12 rounded-xl bg-yellow-400 border-3 border-black flex items-center justify-center font-bebas text-2xl text-black hard-shadow-sm">
            P1
          </div>
          <div>
            <div class="flex items-center gap-1.5">
              <span class="text-[11px] font-black uppercase text-slate-500 tracking-wider">Player 1 (Gold)</span>
              <span id="p1YouTag" class="text-[9px] font-black px-1.5 py-0.5 bg-black text-yellow-400 rounded uppercase">YOU</span>
            </div>
            <div class="flex items-baseline gap-2">
              <span id="scoreP1" class="font-bebas text-4xl md:text-5xl text-black leading-none">0</span>
              <span class="text-xs font-extrabold text-slate-600">REPS</span>
              <span id="leadP1" class="px-2 py-0.5 rounded text-[10px] font-black uppercase bg-yellow-100 text-black border border-black">TIED</span>
            </div>
          </div>
        </div>

        <!-- Dynamic Tug-of-War Dominance Bar (Center) -->
        <div class="flex-1 w-full">
          <div class="flex justify-between text-xs font-black uppercase tracking-wider mb-1 px-1">
            <span class="text-black flex items-center gap-1"><i class="fa-solid fa-crown text-yellow-500"></i> P1 DOMINANCE</span>
            <span id="matchClock" class="font-mono text-xs md:text-sm font-bold bg-black text-yellow-400 px-3 py-0.5 rounded-md border border-black hard-shadow-sm">01:00.0</span>
            <span class="text-red-600 flex items-center gap-1">P2 DOMINANCE <i class="fa-solid fa-crown text-red-600"></i></span>
          </div>

          <!-- The Animated Tug-of-War Bar -->
          <div class="w-full h-10 md:h-12 bg-black border-3 border-black rounded-2xl overflow-hidden relative flex hard-shadow">
            <!-- P1 Yellow Segment -->
            <div id="barP1" class="h-full bg-yellow-400 transition-all duration-300 ease-out flex items-center justify-start pl-3 text-black font-bebas text-lg md:text-xl font-bold whitespace-nowrap overflow-hidden" style="width: 50%">
              <span id="pctP1">50%</span>
            </div>
            <!-- P2 Red Segment -->
            <div id="barP2" class="h-full bg-red-600 transition-all duration-300 ease-out flex items-center justify-end pr-3 text-white font-bebas text-lg md:text-xl font-bold whitespace-nowrap overflow-hidden" style="width: 50%">
              <span id="pctP2">50%</span>
            </div>
            <!-- Sliding VS Center Pin -->
            <div id="pinVS" class="absolute top-0 bottom-0 -translate-x-1/2 flex items-center justify-center transition-all duration-300 ease-out pointer-events-none z-10" style="left: 50%">
              <div class="px-2.5 py-1 bg-black text-yellow-400 border-2 border-white rounded-lg font-bebas text-xs md:text-sm tracking-wider hard-shadow-sm flex items-center gap-1 shadow-md">
                <i class="fa-solid fa-bolt text-[10px] text-yellow-400 animate-pulse"></i>
                <span>VS</span>
              </div>
            </div>
          </div>
        </div>

        <!-- P2 Player Badge (Right) -->
        <div class="flex items-center gap-3 min-w-[220px] justify-end w-full md:w-auto">
          <div class="text-right">
            <div class="flex items-center gap-1.5 justify-end">
              <span id="p2YouTag" class="text-[9px] font-black px-1.5 py-0.5 bg-red-600 text-white rounded uppercase hidden">YOU</span>
              <span class="text-[11px] font-black uppercase text-slate-500 tracking-wider">Player 2 (Red)</span>
            </div>
            <div class="flex items-baseline gap-2 justify-end">
              <span id="leadP2" class="px-2 py-0.5 rounded text-[10px] font-black uppercase bg-red-100 text-red-700 border border-black">TIED</span>
              <span id="scoreP2" class="font-bebas text-4xl md:text-5xl text-red-600 leading-none">0</span>
              <span class="text-xs font-extrabold text-slate-600">REPS</span>
            </div>
          </div>
          <div class="w-12 h-12 rounded-xl bg-red-600 border-3 border-black flex items-center justify-center font-bebas text-2xl text-white hard-shadow-sm">
            P2
          </div>
        </div>

      </div>
    </div>

    <!-- ==================== YOUR FULL CAMERA VIEWPORT ==================== -->
    <div class="comic-card p-3 sm:p-4 bg-white hard-shadow-lg">
      <div class="relative w-full h-[66vh] min-h-[480px] max-h-[820px] bg-slate-950 rounded-2xl overflow-hidden border-3 border-black hard-shadow flex items-center justify-center">

        <!-- Hidden Video & Canvas -->
        <video id="webcam" autoplay playsinline muted class="hidden"></video>
        <canvas id="outputCanvas" class="w-full h-full object-contain bg-slate-950"></canvas>

        <!-- Top Floating Status Pill -->
        <div id="hudPill" class="absolute top-4 left-1/2 -translate-x-1/2 px-5 py-2 rounded-xl bg-black/95 text-white border-2 border-yellow-400 text-xs md:text-sm font-extrabold tracking-wide hard-shadow-sm flex items-center gap-2 z-10 text-center">
          <span id="hudDot" class="w-2.5 h-2.5 rounded-full bg-yellow-400 animate-ping"></span>
          <span id="hudPillText">Click START to ready up &bull; Battle starts when both click Start!</span>
        </div>

        <!-- Role Badge in Top-Left of Video -->
        <div id="badgeRoleIndicator" class="absolute top-4 left-4 px-3.5 py-1.5 rounded-xl bg-yellow-400 text-black border-2 border-black font-bebas text-sm md:text-base tracking-wider hard-shadow-sm z-10 pointer-events-none">
          🟡 YOUR CAMERA &bull; PLAYER 1 (GOLD)
        </div>

        <!-- Opponent Live Status in Top-Right of Video -->
        <div class="absolute top-4 right-4 px-3.5 py-1.5 rounded-xl bg-black/90 text-white border-2 border-white flex items-center gap-2 text-xs font-black hard-shadow-sm z-10 pointer-events-none">
          <span class="text-slate-300">OPPONENT:</span>
          <span id="oppRepsText" class="text-yellow-400 font-bebas text-base">0 REPS</span>
          <span id="oppReadyPill" class="text-[9px] px-2 py-0.5 rounded font-black uppercase bg-slate-700 text-slate-300 border border-slate-600">NOT READY</span>
        </div>

        <!-- 10-Second Synchronized Preparation Countdown Modal -->
        <div id="countdownOverlay" class="hidden absolute inset-0 bg-black/85 backdrop-blur-sm flex flex-col items-center justify-center p-6 text-center z-30">
          <div class="p-6 md:p-8 rounded-3xl bg-yellow-400 border-4 border-black hard-shadow-lg max-w-md w-full flex flex-col items-center">
            <span class="font-bebas text-2xl text-black tracking-widest uppercase">BATTLE BEGINNING!</span>
            <div class="my-4 w-32 h-32 rounded-full bg-white border-4 border-black flex items-center justify-center hard-shadow-sm">
              <span id="countdownNumber" class="font-bebas text-7xl text-black leading-none">10</span>
            </div>
            <h3 class="text-xl font-black text-black mb-1">Both Players Ready!</h3>
            <p class="text-xs font-bold text-black/80 mb-4">Step back into side-view plank position! Match begins when countdown reaches zero.</p>
            <div class="w-full h-3 bg-white border-2 border-black rounded-full overflow-hidden">
              <div id="countdownBar" class="h-full bg-black transition-all duration-200" style="width: 100%"></div>
            </div>
          </div>
        </div>

        <!-- Victory / Knockout Modal -->
        <div id="winnerOverlay" class="hidden absolute inset-0 bg-black/90 backdrop-blur-md flex flex-col items-center justify-center p-6 z-40">
          <div class="bg-white border-4 border-black p-6 md:p-8 rounded-3xl max-w-lg w-full text-center hard-shadow-lg">
            <div class="w-20 h-20 mx-auto rounded-3xl bg-yellow-400 border-3 border-black flex items-center justify-center text-4xl mb-4 hard-shadow-sm">
              <i class="fa-solid fa-trophy text-black"></i>
            </div>
            <h2 id="winTitle" class="font-bebas text-5xl md:text-6xl text-black leading-none mb-2">PLAYER 1 WINS!</h2>
            <p id="winReason" class="text-sm font-extrabold uppercase tracking-wider text-slate-600 mb-6">DOMINANCE KNOCKOUT</p>

            <div class="grid grid-cols-2 gap-4 mb-6">
              <div class="p-4 bg-yellow-100/70 border-3 border-black rounded-2xl hard-shadow-sm">
                <div class="text-xs font-black uppercase text-black">P1 Score</div>
                <div id="winP1Reps" class="font-bebas text-5xl text-black leading-none mt-1">0</div>
                <div class="text-[10px] font-bold text-slate-600 uppercase mt-1">Strict Reps</div>
              </div>
              <div class="p-4 bg-red-100/70 border-3 border-black rounded-2xl hard-shadow-sm">
                <div class="text-xs font-black uppercase text-red-600">P2 Score</div>
                <div id="winP2Reps" class="font-bebas text-5xl text-red-600 leading-none mt-1">0</div>
                <div class="text-[10px] font-bold text-slate-600 uppercase mt-1">Strict Reps</div>
              </div>
            </div>

            <button onclick="toggleReady()" class="w-full py-4 bg-yellow-400 text-black font-extrabold text-lg uppercase tracking-wider rounded-2xl border-3 border-black hard-shadow hard-button transition flex items-center justify-center gap-2">
              <i class="fa-solid fa-rotate-right"></i>
              <span>START REMATCH</span>
            </button>
          </div>
        </div>

      </div>
    </div>

    <!-- ==================== STREAMLINED BOTTOM CONTROLS ==================== -->
    <div class="comic-card p-3 sm:p-4 bg-white hard-shadow">
      <div class="flex flex-col sm:flex-row items-center justify-between gap-3">
        
        <!-- Ready / Start Button -->
        <button id="btnReady" onclick="toggleReady()" class="w-full sm:w-2/3 py-4 bg-yellow-400 hover:bg-yellow-300 text-black font-extrabold text-base md:text-lg uppercase tracking-wider rounded-2xl border-3 border-black hard-shadow hard-button flex items-center justify-center gap-3 transition">
          <i class="fa-solid fa-play"></i>
          <span id="btnReadyText">START / READY UP</span>
        </button>

        <!-- Stop Match Button -->
        <button id="btnStop" onclick="triggerBattleStop()" class="w-full sm:w-1/3 py-4 bg-slate-200 text-slate-600 hover:bg-red-600 hover:text-white font-extrabold text-base md:text-lg uppercase tracking-wider rounded-2xl border-3 border-black hard-shadow hard-button transition flex items-center justify-center gap-2">
          <i class="fa-solid fa-stop"></i>
          <span>STOP MATCH</span>
        </button>

      </div>
    </div>

  </main>

  <!-- Client Script for Multi-Device 1v1 Battle -->
  <script>
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('outputCanvas');
    const ctx = canvas.getContext('2d');

    const urlParams = new URLSearchParams(window.location.search);
    let myRole = parseInt(urlParams.get('role') || urlParams.get('p') || '1');
    let isProcessing = false;
    let amIReady = false;
    let lastAudioAnnounce = -1;
    let lastP1Reps = 0;
    let lastP2Reps = 0;
    let lastLeader = "TIE";

    function applyRoleUI() {{
      const b1 = document.getElementById('btnRole1');
      const b2 = document.getElementById('btnRole2');
      const tag1 = document.getElementById('p1YouTag');
      const tag2 = document.getElementById('p2YouTag');
      const badge = document.getElementById('badgeRoleIndicator');

      if (myRole === 1) {{
        b1.className = "px-3 py-1 rounded-lg text-xs font-black uppercase bg-yellow-400 text-black border-2 border-black hard-shadow-sm transition";
        b2.className = "px-3 py-1 rounded-lg text-xs font-black uppercase bg-white text-slate-700 border-2 border-black hover:bg-red-500 hover:text-white transition";
        tag1.classList.remove('hidden');
        tag2.classList.add('hidden');
        badge.className = "absolute top-4 left-4 px-3.5 py-1.5 rounded-xl bg-yellow-400 text-black border-2 border-black font-bebas text-sm md:text-base tracking-wider hard-shadow-sm z-10 pointer-events-none";
        badge.innerText = "🟡 YOUR CAMERA • PLAYER 1 (GOLD)";
      }} else {{
        b2.className = "px-3 py-1 rounded-lg text-xs font-black uppercase bg-red-600 text-white border-2 border-black hard-shadow-sm transition";
        b1.className = "px-3 py-1 rounded-lg text-xs font-black uppercase bg-white text-slate-700 border-2 border-black hover:bg-yellow-400 hover:text-black transition";
        tag2.classList.remove('hidden');
        tag1.classList.add('hidden');
        badge.className = "absolute top-4 left-4 px-3.5 py-1.5 rounded-xl bg-red-600 text-white border-2 border-black font-bebas text-sm md:text-base tracking-wider hard-shadow-sm z-10 pointer-events-none";
        badge.innerText = "🔴 YOUR CAMERA • PLAYER 2 (RED)";
      }}
    }}

    function switchRole(newRole) {{
      myRole = newRole;
      amIReady = false;
      const url = new URL(window.location);
      url.searchParams.set('role', newRole);
      window.history.pushState({{}}, '', url);
      applyRoleUI();
      updateReadyButtonUI(false);
    }}

    function copyOpponentLink() {{
      const oppRole = myRole === 1 ? 2 : 1;
      const url = new URL(window.location.origin + '/battle');
      url.searchParams.set('role', oppRole);
      navigator.clipboard.writeText(url.href).then(() => {{
        alert("Copied Player " + oppRole + " link to clipboard!\\n\\nShare it with your friend or open it in a second window:\\n" + url.href);
      }});
    }}

    applyRoleUI();

    navigator.mediaDevices.getUserMedia({{ video: {{ width: 1280, height: 720 }} }})
      .then(stream => {{
        video.srcObject = stream;
        video.play();
        requestAnimationFrame(processLoop);
      }})
      .catch(err => {{
        navigator.mediaDevices.getUserMedia({{ video: {{ width: 640, height: 480 }} }})
          .then(stream => {{
            video.srcObject = stream;
            video.play();
            requestAnimationFrame(processLoop);
          }})
          .catch(e => {{
            document.getElementById('hudPillText').innerText = "Webcam permission needed for battle.";
          }});
      }});

    async function processLoop() {{
      if (video.readyState === 4 && !isProcessing) {{
        isProcessing = true;
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.55);

        try {{
          const res = await fetch('/api/battle/frame_player', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ player: myRole, image: dataUrl }})
          }});
          const data = await res.json();
          if (!data.error) {{
            updateBattleDashboard(data);
            drawSinglePlayerCanvas(data);
          }}
        }} catch (e) {{
          console.error(e);
        }} finally {{
          isProcessing = false;
        }}
      }}
      requestAnimationFrame(processLoop);
    }}

    function speak(text) {{
      if ('speechSynthesis' in window) {{
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.25;
        window.speechSynthesis.speak(utter);
      }}
    }}

    function updateReadyButtonUI(ready) {{
      const btn = document.getElementById('btnReady');
      const txt = document.getElementById('btnReadyText');
      if (ready) {{
        btn.className = "w-full sm:w-2/3 py-4 bg-emerald-500 hover:bg-emerald-400 text-white font-extrabold text-base md:text-lg uppercase tracking-wider rounded-2xl border-3 border-black hard-shadow hard-button flex items-center justify-center gap-3 transition";
        txt.innerText = "READY ✓ (WAITING FOR OPPONENT...)";
      }} else {{
        btn.className = "w-full sm:w-2/3 py-4 bg-yellow-400 hover:bg-yellow-300 text-black font-extrabold text-base md:text-lg uppercase tracking-wider rounded-2xl border-3 border-black hard-shadow hard-button flex items-center justify-center gap-3 transition";
        txt.innerText = "START / READY UP";
      }}
    }}

    function updateBattleDashboard(data) {{
      const badgeMode = document.getElementById('badgeMode');
      if (badgeMode) badgeMode.innerText = data.mode;

      document.getElementById('matchClock').innerText = data.timer_str;

      // Tug-of-War Dominance Bar increases and decreases dynamically!
      const p1Dom = data.p1_dominance;
      const p2Dom = data.p2_dominance;
      document.getElementById('barP1').style.width = `${{p1Dom}}%`;
      document.getElementById('barP2').style.width = `${{p2Dom}}%`;
      document.getElementById('pctP1').innerText = `${{Math.round(p1Dom)}}%`;
      document.getElementById('pctP2').innerText = `${{Math.round(p2Dom)}}%`;
      document.getElementById('pinVS').style.left = `${{p1Dom}}%`;

      document.getElementById('scoreP1').innerText = data.p1_reps;
      document.getElementById('scoreP2').innerText = data.p2_reps;

      // Update Opponent Corner Badge
      const oppReps = myRole === 1 ? data.p2_reps : data.p1_reps;
      const oppReady = myRole === 1 ? data.p2_ready : data.p1_ready;
      document.getElementById('oppRepsText').innerText = `${{oppReps}} REPS`;
      const oppPill = document.getElementById('oppReadyPill');
      if (oppReady) {{
        oppPill.className = "text-[9px] px-2 py-0.5 rounded font-black uppercase bg-emerald-500 text-white border border-black";
        oppPill.innerText = "READY ✓";
      }} else {{
        oppPill.className = "text-[9px] px-2 py-0.5 rounded font-black uppercase bg-slate-700 text-slate-300 border border-slate-600";
        oppPill.innerText = "NOT READY";
      }}

      // Voice announcements
      if (data.mode === 'WORKOUT') {{
        if (data.p1_reps > lastP1Reps) {{
          speak(myRole === 1 ? "Good rep! Point to you!" : "Player 1 scored!");
          lastP1Reps = data.p1_reps;
        }}
        if (data.p2_reps > lastP2Reps) {{
          speak(myRole === 2 ? "Good rep! Point to you!" : "Player 2 scored!");
          lastP2Reps = data.p2_reps;
        }}
        if (data.leader !== lastLeader && data.leader !== 'TIE') {{
          speak(data.leader === (myRole === 1 ? 'P1' : 'P2') ? "You take the lead!" : "Opponent takes the lead!");
          lastLeader = data.leader;
        }}
      }}

      // Lead indicators
      const leadP1 = document.getElementById('leadP1');
      const leadP2 = document.getElementById('leadP2');
      if (data.score_diff > 0) {{
        leadP1.innerText = `+${{data.score_diff}} AHEAD 🔥`;
        leadP1.className = "px-2 py-0.5 rounded text-[10px] font-black uppercase bg-yellow-400 text-black border border-black";
        leadP2.innerText = `-${{data.score_diff}} BEHIND`;
        leadP2.className = "px-2 py-0.5 rounded text-[10px] font-black uppercase bg-slate-100 text-slate-600 border border-black";
      }} else if (data.score_diff < 0) {{
        leadP1.innerText = `${{data.score_diff}} BEHIND`;
        leadP1.className = "px-2 py-0.5 rounded text-[10px] font-black uppercase bg-slate-100 text-slate-600 border border-black";
        leadP2.innerText = `+${{-data.score_diff}} AHEAD 🔥`;
        leadP2.className = "px-2 py-0.5 rounded text-[10px] font-black uppercase bg-red-600 text-white border border-black";
      }} else {{
        leadP1.innerText = "TIED";
        leadP1.className = "px-2 py-0.5 rounded text-[10px] font-black uppercase bg-yellow-100 text-black border border-black";
        leadP2.innerText = "TIED";
        leadP2.className = "px-2 py-0.5 rounded text-[10px] font-black uppercase bg-red-100 text-red-700 border border-black";
      }}

      // Floating HUD Message based on readiness
      const hudPillText = document.getElementById('hudPillText');
      if (data.mode === 'IDLE') {{
        const myReady = myRole === 1 ? data.p1_ready : data.p2_ready;
        updateReadyButtonUI(myReady);
        if (!myReady && !oppReady) {{
          hudPillText.innerText = "Click START to ready up • Game begins when both click Start!";
        }} else if (myReady && !oppReady) {{
          hudPillText.innerText = "READY! Waiting for Opponent to click Start...";
        }} else if (!myReady && oppReady) {{
          hudPillText.innerText = "Opponent is READY! Click Start to begin battle!";
        }}
      }} else if (data.mode === 'WORKOUT') {{
        hudPillText.innerText = "BATTLE IN PROGRESS! Push! Complete full reps!";
      }}

      // Countdown Modal
      const cdOverlay = document.getElementById('countdownOverlay');
      if (data.mode === 'COUNTDOWN') {{
        cdOverlay.classList.remove('hidden');
        const remInt = Math.ceil(data.countdown_rem);
        document.getElementById('countdownNumber').innerText = remInt;
        document.getElementById('countdownBar').style.width = `${{((10.0 - data.countdown_rem) / 10.0) * 100}}%`;
        if (remInt <= 3 && remInt > 0 && remInt !== lastAudioAnnounce) {{
          lastAudioAnnounce = remInt;
          speak(String(remInt));
        }}
      }} else {{
        if (!cdOverlay.classList.contains('hidden') && lastAudioAnnounce !== -1) {{
          speak("FIGHT!");
          lastAudioAnnounce = -1;
        }}
        cdOverlay.classList.add('hidden');
      }}

      // Winner Modal
      const winModal = document.getElementById('winnerOverlay');
      if (data.winner) {{
        document.getElementById('winTitle').innerText = data.winner === 'DRAW' ? "DRAW MATCH!" : `${{data.winner === 'P1' ? 'PLAYER 1' : 'PLAYER 2'}} WINS!`;
        document.getElementById('winReason').innerText = data.win_reason;
        document.getElementById('winP1Reps').innerText = data.p1_reps;
        document.getElementById('winP2Reps').innerText = data.p2_reps;
        winModal.classList.remove('hidden');
      }} else {{
        winModal.classList.add('hidden');
      }}
    }}

    function drawSinglePlayerCanvas(data) {{
      if (!data.my_result || !data.my_result.landmarks || data.my_result.landmarks.length < 5) return;
      const pts = data.my_result.landmarks.map(p => ({{ x: p.x * canvas.width, y: p.y * canvas.height }}));
      const [sh, el, wr, hip, ank] = pts;

      const myColor = myRole === 1 ? '#FACC15' : '#EF4444';
      const isGood = data.my_result.is_good_form;
      const coreColor = isGood ? myColor : '#EF4444';

      // 1. Thick dark stroke outline
      ctx.lineWidth = 10;
      ctx.strokeStyle = '#000000';
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(sh.x, sh.y); ctx.lineTo(el.x, el.y); ctx.lineTo(wr.x, wr.y);
      ctx.moveTo(sh.x, sh.y); ctx.lineTo(hip.x, hip.y); ctx.lineTo(ank.x, ank.y);
      ctx.stroke();

      // 2. High contrast inner core line
      ctx.lineWidth = 6;
      ctx.strokeStyle = coreColor;
      ctx.beginPath();
      ctx.moveTo(sh.x, sh.y); ctx.lineTo(el.x, el.y); ctx.lineTo(wr.x, wr.y);
      ctx.moveTo(sh.x, sh.y); ctx.lineTo(hip.x, hip.y); ctx.lineTo(ank.x, ank.y);
      ctx.stroke();

      // 3. Highlighted Joints
      pts.forEach(p => {{
        ctx.fillStyle = '#000000';
        ctx.beginPath(); ctx.arc(p.x, p.y, 9, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = coreColor;
        ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, Math.PI * 2); ctx.fill();
      }});

      // 4. Live Posture Angle Badges directly on Joints
      function drawBadge(x, y, text, subtext, bgColor, textColor) {{
        ctx.save();
        ctx.font = 'bold 15px monospace';
        const tWidth = ctx.measureText(text).width;
        const bWidth = Math.max(tWidth + 28, 76);
        const bHeight = subtext ? 36 : 24;
        const bx = Math.min(Math.max(x - bWidth / 2, 12), canvas.width - bWidth - 12);
        const by = Math.min(Math.max(y - 42, 12), canvas.height - bHeight - 12);

        ctx.fillStyle = '#000000';
        ctx.fillRect(bx + 2, by + 2, bWidth, bHeight);
        ctx.fillStyle = bgColor;
        ctx.fillRect(bx, by, bWidth, bHeight);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#000000';
        ctx.strokeRect(bx, by, bWidth, bHeight);

        ctx.fillStyle = textColor;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (subtext) {{
          ctx.font = '900 14px monospace';
          ctx.fillText(text, bx + bWidth / 2, by + 13);
          ctx.font = 'bold 9px sans-serif';
          ctx.fillText(subtext, bx + bWidth / 2, by + 26);
        }} else {{
          ctx.font = '900 14px monospace';
          ctx.fillText(text, bx + bWidth / 2, by + bHeight / 2);
        }}
        ctx.restore();
      }}

      if (data.my_result.elbow_angle !== undefined) {{
        const elDeg = Math.round(data.my_result.elbow_angle);
        const elBg = elDeg <= 90 ? '#22C55E' : (elDeg <= 120 ? '#FACC15' : '#FFFFFF');
        const elSub = elDeg <= 90 ? 'DEPTH OK' : (elDeg > 150 ? 'LOCKED' : 'LOWER');
        drawBadge(el.x, el.y, `${{elDeg}}°`, elSub, elBg, '#000000');
      }}

      if (data.my_result.body_angle !== undefined) {{
        const bdDeg = Math.round(data.my_result.body_angle);
        const bdBg = bdDeg >= 160 ? '#22C55E' : '#EF4444';
        const bdSub = bdDeg >= 160 ? 'SPINE OK' : 'SAG / PIKE';
        const txtCol = bdDeg >= 160 ? '#000000' : '#FFFFFF';
        drawBadge(hip.x, hip.y, `${{bdDeg}}°`, bdSub, bdBg, txtCol);
      }}
    }}

    async function toggleReady() {{
      document.getElementById('winnerOverlay').classList.add('hidden');
      amIReady = !amIReady;
      updateReadyButtonUI(amIReady);
      await fetch('/api/battle/ready', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ player: myRole, ready: amIReady }})
      }});
      if (amIReady) {{
        speak("Ready! Waiting for opponent");
      }}
    }}

    async function triggerBattleStop() {{
      amIReady = false;
      updateReadyButtonUI(false);
      await fetch('/api/battle/stop', {{ method: 'POST' }});
      speak("Battle stopped");
    }}

    window.addEventListener('keydown', (e) => {{
      if (e.key === 's' || e.key === 'S') toggleReady();
      if (e.key === ' ' || e.key === 'x' || e.key === 'X') triggerBattleStop();
    }});
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run("app_web:app", host="127.0.0.1", port=8000, reload=False)
