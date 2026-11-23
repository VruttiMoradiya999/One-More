"""
One-More: Strict AI Pushup Counter & Form Coach.

Features:
  - Modern Interactive UI with Start & Stop buttons (mouse-clickable & keyboard shortcuts).
  - 10-Second Preparation Countdown upon clicking Start.
  - Workout Timer that starts automatically after the 10-second countdown.
  - Strict Form Validation:
      * Initial 1-second plank hold before counting begins.
      * Full depth: elbow angle <= 90 degrees.
      * Full lockout: elbow angle >= 160 degrees.
      * Strict body straightness: shoulder-hip-ankle angle >= 160 degrees (no piking or sagging).
      * Cadence check: minimum rep duration >= 0.8s to reject bouncing/jitter.
      * Real-time audio & visual guidance for form corrections.

Controls:
  - Click [START] button or press 'S' to begin 10s preparation countdown.
  - Click [STOP] button or press 'Space' / 'X' to finish workout and see summary.
  - Press 'Q' or ESC to quit.
"""


import math
import os
import subprocess
import threading
import time
from collections import deque
from enum import Enum

import cv2
import numpy as np
import mediapipe as mp

# ============================== CONFIGURATION ==============================
CANVAS_W, CANVAS_H = 1280, 720
VIEWPORT_X, VIEWPORT_Y = 20, 20
VIEWPORT_W, VIEWPORT_H = 860, 680

SIDEBAR_X = 900
SIDEBAR_W = 360

# Pushup biomechanics thresholds
ELBOW_DOWN = 90          # Max elbow angle at bottom for full depth (degrees)
ELBOW_UP = 160           # Min elbow angle at top for full lockout (degrees)
ELBOW_START = 150        # Angle below which a repetition is considered started
BODY_MIN = 160           # Min shoulder-hip-ankle angle (straight body line)
MAX_TILT_DEG = 32        # Max deviation of body from horizontal (degrees)
HANDS_UNDER_SHOULDER = 0.35  # Wrist-shoulder horizontal offset tolerance
MIN_VISIBILITY = 0.6     # Min confidence for landmark tracking

COUNTDOWN_DURATION = 10.0   # 10 seconds preparation countdown after clicking Start
READY_HOLD_SEC = 1.0        # Hold plank position still before starting count
LOST_POSITION_SEC = 0.7     # Leaving position cancels current rep
CONFIRM_FRAMES = 3          # Consecutive frames required to confirm angle threshold
MIN_REP_SEC = 0.8           # Faster than this = jitter/bounce
MAX_REP_SEC = 15.0          # Slower than this = rep cancelled
SMOOTH_FRAMES = 5           # Moving average window to remove jitter
# ===========================================================================

# MediaPipe Initialization
try:
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    P = mp_pose.PoseLandmark
except AttributeError:
    raise ImportError(
        "MediaPipe 'solutions' module not found. "
        "Please ensure mediapipe==0.10.14 is installed in your Python 3.11 venv."
    )

SIDES = {
    "left": [P.LEFT_SHOULDER, P.LEFT_ELBOW, P.LEFT_WRIST, P.LEFT_HIP, P.LEFT_ANKLE],
    "right": [P.RIGHT_SHOULDER, P.RIGHT_ELBOW, P.RIGHT_WRIST, P.RIGHT_HIP, P.RIGHT_ANKLE],
}


class AppMode(Enum):
    IDLE = 1         # Camera preview active, waiting for user to click START
    COUNTDOWN = 2    # 10-second preparation countdown
    WORKOUT = 3      # Active workout, timer running, counting strict pushups
    STOPPED = 4      # Workout ended, summary displayed


# ============================== AUDIO FEEDBACK ==============================
def play_sound_async(sound_name=None, speech_text=None):
    """Play macOS system sound or speech in a non-blocking daemon thread."""
    def _worker():
        try:
            if sound_name:
                sound_path = f"/System/Library/Sounds/{sound_name}.aiff"
                if os.path.exists(sound_path):
                    subprocess.run(["afplay", sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                elif speech_text:
                    subprocess.run(["say", "-r", "220", speech_text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif speech_text:
                subprocess.run(["say", "-r", "220", speech_text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


# ========================== BIOMECHANICS FUNCTIONS ==========================
def calc_angle(a, b, c):
    """Calculate the interior angle at point b (degrees) formed by a-b-c."""
    v1, v2 = a - b, c - b
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 * norm2 < 1e-7:
        return 180.0
    cos = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def choose_best_side(lm):
    """Select the body side with highest tracking visibility."""
    scores = {s: min(lm[i].visibility for i in ids) for s, ids in SIDES.items()}
    return max(scores, key=scores.get)


def is_plank_geometry_ok(sh, wr, ank):
    """Check if the user is horizontal with hands under shoulders."""
    dx, dy = abs(sh[0] - ank[0]), abs(sh[1] - ank[1])
    tilt = np.degrees(np.arctan2(dy, dx + 1e-9))
    body_len = np.hypot(dx, dy) + 1e-9
    horizontal = tilt <= MAX_TILT_DEG
    hands_under = abs(wr[0] - sh[0]) <= HANDS_UNDER_SHOULDER * body_len
    return bool(horizontal and hands_under)


def diagnose_hip_problem(sh, hip, ank):
    """Determine whether hips are piking (too high) or sagging (too low)."""
    if abs(ank[0] - sh[0]) < 1e-6:
        return "Keep body straight"
    t = (hip[0] - sh[0]) / (ank[0] - sh[0])
    line_y = sh[1] + t * (ank[1] - sh[1])
    return "Hips too high" if hip[1] < line_y else "Hips sagging"


# ============================== UI PRIMITIVES ==============================
def draw_rounded_rect(img, x, y, w, h, color, radius=12, thickness=-1):
    """Draw a smooth rounded rectangle with anti-aliasing."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    radius = max(2, min(radius, w // 2, h // 2))
    x2, y2 = x + w, y + h
    if thickness == -1:
        cv2.rectangle(img, (x + radius, y), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x, y + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x + radius, y + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x + radius, y), (x2 - radius, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y + radius), (x, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)


def draw_card(img, x, y, w, h, bg_color=(255, 255, 255), border_color=(17, 17, 17), radius=10):
    """Draw a comic-style container card with bold black border."""
    draw_rounded_rect(img, x, y, w, h, bg_color, radius=radius, thickness=-1)
    draw_rounded_rect(img, x, y, w, h, border_color, radius=radius, thickness=2)


def draw_button(img, x, y, w, h, text, base_color, hover_color, is_hovered, is_active=True, icon=""):
    """Draw an interactive comic neo-brutalist button."""
    if not is_active:
        bg = (220, 220, 220)
        border = (160, 160, 160)
        text_color = (130, 130, 130)
    elif is_hovered:
        bg = hover_color
        border = (17, 17, 17)
        text_color = (0, 0, 0) if (base_color[0] > 100 or base_color[1] > 100) else (255, 255, 255)
    else:
        bg = base_color
        border = (17, 17, 17)
        text_color = (0, 0, 0) if (base_color[0] > 100 or base_color[1] > 100) else (255, 255, 255)

    draw_rounded_rect(img, x, y, w, h, bg, radius=12, thickness=-1)
    draw_rounded_rect(img, x, y, w, h, border, radius=12, thickness=3 if is_hovered else 2)

    full_text = f"{icon}  {text}".strip() if icon else text
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.65
    thick = 2
    (tw, th), _ = cv2.getTextSize(full_text, font, scale, thick)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(img, full_text, (tx, ty), font, scale, text_color, thick, cv2.LINE_AA)


def draw_progress_bar(img, x, y, w, h, ratio, color, bg_color=(45, 38, 34), radius=4):
    """Draw a smooth horizontal progress bar."""
    draw_rounded_rect(img, x, y, w, h, bg_color, radius=radius, thickness=-1)
    fill_w = max(4, int(w * np.clip(ratio, 0.0, 1.0)))
    if fill_w > radius * 2:
        draw_rounded_rect(img, x, y, fill_w, h, color, radius=radius, thickness=-1)
    else:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)


# ============================== MAIN APPLICATION ==============================
class PushupApp:
    def __init__(self):
        self.mode = AppMode.IDLE
        self.countdown_start = 0.0
        self.workout_start = 0.0
        self.workout_duration = 0.0

        # Rep counters
        self.good_reps = 0
        self.bad_reps = 0

        # Rep state machine (inside WORKOUT mode)
        self.state = "WAIT"   # WAIT -> UP -> LOWERING -> DOWN -> UP ...
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

        self.elbow_angle = None
        self.body_angle = None
        self.status_msg = "Can't see full body"

        self.feedback = ""
        self.feedback_time = 0.0
        self.last_audio_countdown_sec = -1

        # Interactive UI buttons
        self.btn_start = (SIDEBAR_X, 95, SIDEBAR_W, 52)
        self.btn_stop = (SIDEBAR_X, 158, SIDEBAR_W, 52)

        self.mouse_pos = (0, 0)
        self.mouse_clicked = False

    def on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.handle_click(x, y)

    def is_inside(self, pt, rect):
        rx, ry, rw, rh = rect
        px, py = pt
        return rx <= px <= rx + rw and ry <= py <= ry + rh

    def handle_click(self, x, y):
        pt = (x, y)
        # Click START button
        if self.is_inside(pt, self.btn_start):
            if self.mode in (AppMode.IDLE, AppMode.STOPPED):
                self.start_countdown()

        # Click STOP button
        elif self.is_inside(pt, self.btn_stop):
            if self.mode in (AppMode.COUNTDOWN, AppMode.WORKOUT):
                self.stop_workout()

    def start_countdown(self):
        self.mode = AppMode.COUNTDOWN
        self.countdown_start = time.time()
        self.good_reps = 0
        self.bad_reps = 0
        self.feedback = ""
        self.last_audio_countdown_sec = -1
        self.reset_rep_machine()
        play_sound_async("Tink", "Get ready")

    def stop_workout(self):
        if self.mode == AppMode.WORKOUT:
            self.workout_duration = time.time() - self.workout_start
        else:
            self.workout_duration = 0.0
        self.mode = AppMode.STOPPED
        self.state = "WAIT"
        play_sound_async("Glass", "Workout stopped")

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

    # -------------------------- WORKOUT LOGIC --------------------------
    def update_workout_logic(self, valid, sh, hip, ank, elbow, body, now):
        if not valid:
            self.hold_start = None
            if self.lost_since is None:
                self.lost_since = now
            if self.state != "WAIT" and (now - self.lost_since) > LOST_POSITION_SEC:
                if self.state in ("LOWERING", "DOWN"):
                    self.feedback = "Rep cancelled: left position"
                    self.feedback_time = now
                    play_sound_async("Basso", "Rep cancelled")
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
                    play_sound_async("Hero", "Ready")
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
            # Check body alignment throughout the rep
            if body < BODY_MIN:
                self.body_bad_cnt += 1
            else:
                self.body_bad_cnt = 0

            if self.body_bad_cnt >= CONFIRM_FRAMES and self.body_ok:
                self.body_ok = False
                self.problem = diagnose_hip_problem(sh, hip, ank)

            # Check if rep is too slow
            if now - self.rep_start_time > MAX_REP_SEC:
                self.feedback = "No point: too slow"
                self.feedback_time = now
                self.state = "UP"
                self.reset_rep_machine()
                play_sound_async("Basso", "Too slow")

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
                        play_sound_async("Basso", "Go lower")

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
                        play_sound_async("Basso", problems[0])
                    else:
                        self.good_reps += 1
                        self.feedback = f"GOOD REP! (#{self.good_reps})"
                        play_sound_async("Ping", str(self.good_reps))

                    self.feedback_time = now
                    self.state = "UP"
                    self.reset_rep_machine()

    # -------------------------- DRAWING HUD --------------------------
    def draw_skeleton(self, frame, lm, w, h, is_good_form):
        """Draw aesthetic joints and bones on camera feed."""
        if not self.side or self.side not in SIDES:
            return
        ids = SIDES[self.side]
        points = [np.array([int(lm[i].x * w), int(lm[i].y * h)]) for i in ids]
        sh, el, wr, hip, ank = points

        bone_color = (60, 220, 110) if is_good_form else (50, 70, 240)
        thick = 3

        # Draw bone connections
        cv2.line(frame, tuple(sh), tuple(el), bone_color, thick, cv2.LINE_AA)
        cv2.line(frame, tuple(el), tuple(wr), bone_color, thick, cv2.LINE_AA)
        cv2.line(frame, tuple(sh), tuple(hip), bone_color, thick, cv2.LINE_AA)
        cv2.line(frame, tuple(hip), tuple(ank), bone_color, thick, cv2.LINE_AA)

        # Draw joints
        for pt in points:
            cv2.circle(frame, tuple(pt), 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pt), 5, bone_color, -1, cv2.LINE_AA)

        # Draw elbow angle arc / text
        if self.elbow_angle is not None:
            cv2.putText(frame, f"{int(self.elbow_angle)} deg", (el[0] + 12, el[1] - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw body angle text
        if self.body_angle is not None:
            cv2.putText(frame, f"{int(self.body_angle)} deg", (hip[0] + 12, hip[1] - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def render_sidebar(self, canvas, now):
        """Render the complete right-hand control dashboard."""
        # 1. Header Card
        draw_card(canvas, SIDEBAR_X, 20, SIDEBAR_W, 64, bg_color=(255, 255, 255), border_color=(17, 17, 17))
        cv2.putText(canvas, "REX.FITNESS", (SIDEBAR_X + 16, 46), cv2.FONT_HERSHEY_DUPLEX, 0.72, (17, 17, 17), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Strict Pushup AI Coach", (SIDEBAR_X + 16, 68), cv2.FONT_HERSHEY_DUPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)

        # Status badge in header
        status_text = {
            AppMode.IDLE: "IDLE",
            AppMode.COUNTDOWN: "PREP",
            AppMode.WORKOUT: "ACTIVE",
            AppMode.STOPPED: "FINISHED",
        }[self.mode]
        badge_color = {
            AppMode.IDLE: (220, 220, 220),
            AppMode.COUNTDOWN: (21, 204, 250),
            AppMode.WORKOUT: (21, 204, 250),
            AppMode.STOPPED: (17, 17, 17),
        }[self.mode]
        badge_txt_color = (17, 17, 17) if self.mode != AppMode.STOPPED else (255, 255, 255)
        draw_rounded_rect(canvas, SIDEBAR_X + SIDEBAR_W - 95, 36, 80, 24, badge_color, radius=6, thickness=-1)
        draw_rounded_rect(canvas, SIDEBAR_X + SIDEBAR_W - 95, 36, 80, 24, (17, 17, 17), radius=6, thickness=2)
        cv2.putText(canvas, status_text, (SIDEBAR_X + SIDEBAR_W - 85, 52), cv2.FONT_HERSHEY_DUPLEX, 0.42, badge_txt_color, 1, cv2.LINE_AA)

        # 2. Buttons: START and STOP
        is_start_hover = self.is_inside(self.mouse_pos, self.btn_start)
        is_stop_hover = self.is_inside(self.mouse_pos, self.btn_stop)

        start_active = self.mode in (AppMode.IDLE, AppMode.STOPPED)
        stop_active = self.mode in (AppMode.COUNTDOWN, AppMode.WORKOUT)

        start_text = "START (10s Prep)" if self.mode != AppMode.WORKOUT else "WORKOUT RUNNING"
        draw_button(
            canvas, *self.btn_start,
            text=start_text,
            base_color=(21, 204, 250),
            hover_color=(50, 220, 255),
            is_hovered=is_start_hover,
            is_active=start_active,
            icon=">"
        )

        draw_button(
            canvas, *self.btn_stop,
            text="STOP WORKOUT",
            base_color=(17, 17, 17),
            hover_color=(45, 45, 45),
            is_hovered=is_stop_hover,
            is_active=stop_active,
            icon="X"
        )

        # 3. Session Timer Card
        draw_card(canvas, SIDEBAR_X, 222, SIDEBAR_W, 90, bg_color=(17, 17, 17), border_color=(17, 17, 17))
        cv2.putText(canvas, "SESSION TIMER", (SIDEBAR_X + 16, 244), cv2.FONT_HERSHEY_DUPLEX, 0.42, (21, 204, 250), 1, cv2.LINE_AA)

        if self.mode == AppMode.IDLE:
            timer_str = "00:00.0"
            timer_color = (180, 180, 180)
            timer_desc = "Click Start (or press S) to begin"
        elif self.mode == AppMode.COUNTDOWN:
            rem = max(0.0, COUNTDOWN_DURATION - (now - self.countdown_start))
            timer_str = f"PREP: {rem:04.1f}s"
            timer_color = (21, 204, 250)
            timer_desc = "Timer starts after 10s countdown"
        elif self.mode == AppMode.WORKOUT:
            elapsed = now - self.workout_start
            m, s = divmod(int(elapsed), 60)
            tenths = int((elapsed - int(elapsed)) * 10)
            timer_str = f"{m:02d}:{s:02d}.{tenths}"
            timer_color = (255, 255, 255)
            timer_desc = "Session in progress - strict reps only"
        else:  # STOPPED
            m, s = divmod(int(self.workout_duration), 60)
            tenths = int((self.workout_duration - int(self.workout_duration)) * 10)
            timer_str = f"{m:02d}:{s:02d}.{tenths}"
            timer_color = (21, 204, 250)
            timer_desc = "Workout finished - click Start to retry"

        cv2.putText(canvas, timer_str, (SIDEBAR_X + 16, 284), cv2.FONT_HERSHEY_DUPLEX, 1.05, timer_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, timer_desc, (SIDEBAR_X + 16, 304), cv2.FONT_HERSHEY_DUPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        # 4. Performance Cards (Good Reps & Bad Reps)
        card_w = (SIDEBAR_W - 14) // 2
        # Good reps (Yellow)
        draw_card(canvas, SIDEBAR_X, 324, card_w, 108, bg_color=(21, 204, 250), border_color=(17, 17, 17))
        cv2.putText(canvas, "STRICT REPS", (SIDEBAR_X + 14, 346), cv2.FONT_HERSHEY_DUPLEX, 0.42, (17, 17, 17), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(self.good_reps), (SIDEBAR_X + 16, 398), cv2.FONT_HERSHEY_DUPLEX, 1.5, (17, 17, 17), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Full depth & lockout", (SIDEBAR_X + 14, 420), cv2.FONT_HERSHEY_DUPLEX, 0.35, (50, 50, 50), 1, cv2.LINE_AA)

        # Bad reps (White + Red)
        draw_card(canvas, SIDEBAR_X + card_w + 14, 324, card_w, 108, bg_color=(255, 255, 255), border_color=(17, 17, 17))
        cv2.putText(canvas, "NO POINT", (SIDEBAR_X + card_w + 28, 346), cv2.FONT_HERSHEY_DUPLEX, 0.42, (30, 30, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(self.bad_reps), (SIDEBAR_X + card_w + 30, 398), cv2.FONT_HERSHEY_DUPLEX, 1.5, (30, 30, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Shallow / bent body", (SIDEBAR_X + card_w + 28, 420), cv2.FONT_HERSHEY_DUPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

        # 5. Form Accuracy Card
        draw_card(canvas, SIDEBAR_X, 444, SIDEBAR_W, 76, bg_color=(255, 255, 255), border_color=(17, 17, 17))
        total_reps = self.good_reps + self.bad_reps
        accuracy = (self.good_reps / total_reps * 100.0) if total_reps > 0 else 100.0
        acc_color = (21, 204, 250) if accuracy >= 70 else (30, 30, 230)

        cv2.putText(canvas, "FORM ACCURACY", (SIDEBAR_X + 16, 468), cv2.FONT_HERSHEY_DUPLEX, 0.42, (17, 17, 17), 1, cv2.LINE_AA)
        acc_text = f"{int(accuracy)}%" if total_reps > 0 else "100%"
        cv2.putText(canvas, acc_text, (SIDEBAR_X + SIDEBAR_W - 75, 468), cv2.FONT_HERSHEY_DUPLEX, 0.65, (17, 17, 17), 2, cv2.LINE_AA)
        draw_progress_bar(canvas, SIDEBAR_X + 16, 482, SIDEBAR_W - 32, 10, accuracy / 100.0, acc_color, bg_color=(230, 230, 230))
        cv2.putText(canvas, f"Total attempts: {total_reps}", (SIDEBAR_X + 16, 508), cv2.FONT_HERSHEY_DUPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

        # 6. Real-Time Angle Gauges Card
        draw_card(canvas, SIDEBAR_X, 532, SIDEBAR_W, 130, bg_color=(255, 255, 255), border_color=(17, 17, 17))
        cv2.putText(canvas, "LIVE BIOMETRIC GAUGES", (SIDEBAR_X + 16, 552), cv2.FONT_HERSHEY_DUPLEX, 0.42, (17, 17, 17), 1, cv2.LINE_AA)

        # Elbow Gauge
        el_val = self.elbow_angle if self.elbow_angle is not None else 180.0
        el_ratio = 1.0 - np.clip((el_val - 60.0) / 120.0, 0.0, 1.0)
        el_color = (21, 204, 250) if el_val <= ELBOW_DOWN else (200, 200, 200)
        cv2.putText(canvas, f"Elbow Depth: {int(el_val)} deg", (SIDEBAR_X + 16, 576), cv2.FONT_HERSHEY_DUPLEX, 0.42, (17, 17, 17), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Goal <= {ELBOW_DOWN} deg", (SIDEBAR_X + SIDEBAR_W - 120, 576), cv2.FONT_HERSHEY_DUPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)
        draw_progress_bar(canvas, SIDEBAR_X + 16, 586, SIDEBAR_W - 32, 8, el_ratio, el_color, bg_color=(230, 230, 230))

        # Body Straightness Gauge
        bd_val = self.body_angle if self.body_angle is not None else 180.0
        bd_ratio = np.clip((bd_val - 120.0) / 60.0, 0.0, 1.0)
        bd_color = (21, 204, 250) if bd_val >= BODY_MIN else (30, 30, 230)
        status_straight = "Straight" if bd_val >= BODY_MIN else "Misaligned"
        cv2.putText(canvas, f"Body Alignment: {int(bd_val)} deg ({status_straight})", (SIDEBAR_X + 16, 622), cv2.FONT_HERSHEY_DUPLEX, 0.42, (17, 17, 17), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Goal >= {BODY_MIN} deg", (SIDEBAR_X + SIDEBAR_W - 120, 622), cv2.FONT_HERSHEY_DUPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)
        draw_progress_bar(canvas, SIDEBAR_X + 16, 632, SIDEBAR_W - 32, 8, bd_ratio, bd_color, bg_color=(230, 230, 230))

        # 7. Hotkey Footer
        cv2.putText(canvas, "[S] Start   •   [Space] Stop   •   [Q] Quit",
                    (SIDEBAR_X + 32, 692), cv2.FONT_HERSHEY_DUPLEX, 0.42, (70, 70, 70), 1, cv2.LINE_AA)

    def render_viewport_overlays(self, frame, now):
        """Render overlays on top of the camera feed (toast feedback, countdown card, summary)."""
        fh, fw = frame.shape[:2]

        # Top floating HUD pill
        hud_bg = (17, 17, 17)
        hud_text = ""
        hud_color = (255, 255, 255)

        if self.mode == AppMode.IDLE:
            hud_text = "STAND BY — CLICK START (10s PREP COUNTDOWN)"
            hud_color = (21, 204, 250)
        elif self.mode == AppMode.COUNTDOWN:
            hud_text = "GET IN POSITION (SIDE VIEW) — 10s PREP ACTIVE"
            hud_color = (21, 204, 250)
        elif self.mode == AppMode.WORKOUT:
            if self.state == "WAIT":
                if self.hold_start is not None:
                    held = min(now - self.hold_start, READY_HOLD_SEC)
                    hud_text = f"HOLD STILL IN PLANK: {held:0.1f} / {READY_HOLD_SEC:.0f}s"
                    hud_color = (21, 204, 250)
                else:
                    hud_text = "HOLD STRAIGHT PLANK FOR 1 SEC TO START"
                    hud_color = (21, 204, 250)
            elif self.state == "UP":
                hud_text = "READY — LOWER YOUR CHEST"
                hud_color = (21, 204, 250)
            elif self.state == "LOWERING":
                hud_text = "GOING DOWN... (REACH <= 90 DEG)"
                hud_color = (255, 255, 255)
            elif self.state == "DOWN":
                hud_text = "FULL DEPTH! NOW PUSH UP!"
                hud_color = (21, 204, 250)
        elif self.mode == AppMode.STOPPED:
            hud_text = "WORKOUT COMPLETE — CHECK SCORECARD"
            hud_color = (21, 204, 250)

        # Draw top HUD pill
        pill_w = 540
        pill_h = 38
        px = (fw - pill_w) // 2
        py = 18
        draw_rounded_rect(frame, px, py, pill_w, pill_h, hud_bg, radius=8, thickness=-1)
        draw_rounded_rect(frame, px, py, pill_w, pill_h, (21, 204, 250), radius=8, thickness=2)

        (tw, th), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_DUPLEX, 0.46, 1)
        cv2.putText(frame, hud_text, (px + (pill_w - tw) // 2, py + (pill_h + th) // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, hud_color, 1, cv2.LINE_AA)

        # Rep toast notification (Good Rep / Bad Rep)
        if self.feedback and (now - self.feedback_time) < 2.0:
            is_good = "GOOD" in self.feedback or "READY" in self.feedback
            toast_bg = (21, 204, 250) if is_good else (30, 30, 220)
            toast_border = (17, 17, 17)
            toast_color = (17, 17, 17) if is_good else (255, 255, 255)

            toast_w = 480
            toast_h = 44
            tx = (fw - toast_w) // 2
            ty = 68
            draw_rounded_rect(frame, tx, ty, toast_w, toast_h, toast_bg, radius=8, thickness=-1)
            draw_rounded_rect(frame, tx, ty, toast_w, toast_h, toast_border, radius=8, thickness=2)

            (tw, th), _ = cv2.getTextSize(self.feedback, cv2.FONT_HERSHEY_DUPLEX, 0.58, 2)
            cv2.putText(frame, self.feedback, (tx + (toast_w - tw) // 2, ty + (toast_h + th) // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.58, toast_color, 2, cv2.LINE_AA)

        # 10-Second Countdown Central Overlay
        if self.mode == AppMode.COUNTDOWN:
            elapsed = now - self.countdown_start
            rem = max(0.0, COUNTDOWN_DURATION - elapsed)
            rem_int = int(math.ceil(rem))

            # Voice announcement for last 3 seconds and Go
            if rem_int in (3, 2, 1) and rem_int != self.last_audio_countdown_sec:
                self.last_audio_countdown_sec = rem_int
                play_sound_async("Tink", str(rem_int))

            if rem <= 0.05:
                # Transition to WORKOUT
                self.mode = AppMode.WORKOUT
                self.workout_start = time.time()
                self.reset_rep_machine()
                play_sound_async("Hero", "Go!")
                return

            # Draw yellow comic countdown card
            cw, ch = 440, 260
            cx = (fw - cw) // 2
            cy = (fh - ch) // 2
            draw_rounded_rect(frame, cx, cy, cw, ch, (21, 204, 250), radius=16, thickness=-1)
            draw_rounded_rect(frame, cx, cy, cw, ch, (17, 17, 17), radius=16, thickness=3)

            cv2.putText(frame, "PREPARATION TIME", (cx + 120, cy + 42), cv2.FONT_HERSHEY_DUPLEX, 0.72, (17, 17, 17), 2, cv2.LINE_AA)
            cv2.putText(frame, "Step back into side-view plank position", (cx + 52, cy + 68), cv2.FONT_HERSHEY_DUPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)

            # Giant countdown number
            num_str = str(rem_int)
            (nw, nh), _ = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_DUPLEX, 3.2, 4)
            cv2.putText(frame, num_str, (cx + (cw - nw) // 2, cy + 172), cv2.FONT_HERSHEY_DUPLEX, 3.2, (17, 17, 17), 4, cv2.LINE_AA)

            # Progress bar
            draw_progress_bar(frame, cx + 40, cy + 208, cw - 80, 12, (COUNTDOWN_DURATION - rem) / COUNTDOWN_DURATION, (17, 17, 17), bg_color=(255, 255, 255))
            cv2.putText(frame, f"Workout timer begins in {rem:.1f}s", (cx + 115, cy + 242), cv2.FONT_HERSHEY_DUPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)

        # Workout Finished Summary Modal
        elif self.mode == AppMode.STOPPED:
            cw, ch = 480, 310
            cx = (fw - cw) // 2
            cy = (fh - ch) // 2
            draw_rounded_rect(frame, cx, cy, cw, ch, (255, 255, 255), radius=16, thickness=-1)
            draw_rounded_rect(frame, cx, cy, cw, ch, (17, 17, 17), radius=16, thickness=3)

            cv2.putText(frame, "WORKOUT COMPLETE!", (cx + 110, cy + 45), cv2.FONT_HERSHEY_DUPLEX, 0.78, (17, 17, 17), 2, cv2.LINE_AA)

            m, s = divmod(int(self.workout_duration), 60)
            dur_str = f"{m:02d}:{s:02d}"
            total = self.good_reps + self.bad_reps
            acc = int(self.good_reps / total * 100) if total > 0 else 100
            pace = (self.good_reps / (self.workout_duration / 60.0)) if self.workout_duration >= 10 else 0.0

            stats = [
                ("Session Duration:", dur_str),
                ("Strict Reps:", f"{self.good_reps} reps"),
                ("No-Point Reps:", f"{self.bad_reps} reps"),
                ("Form Accuracy:", f"{acc}%"),
                ("Workout Cadence:", f"{pace:.1f} reps/min"),
            ]

            sy = cy + 85
            for label, val in stats:
                cv2.putText(frame, label, (cx + 50, sy), cv2.FONT_HERSHEY_DUPLEX, 0.52, (80, 80, 80), 1, cv2.LINE_AA)
                cv2.putText(frame, val, (cx + 310, sy), cv2.FONT_HERSHEY_DUPLEX, 0.52, (17, 17, 17), 2, cv2.LINE_AA)
                sy += 32

            draw_rounded_rect(frame, cx + 40, cy + 252, cw - 80, 38, (21, 204, 250), radius=8, thickness=-1)
            draw_rounded_rect(frame, cx + 40, cy + 252, cw - 80, 38, (17, 17, 17), radius=8, thickness=2)
            cv2.putText(frame, "Click START (or press S) to Restart", (cx + 72, cy + 276), cv2.FONT_HERSHEY_DUPLEX, 0.5, (17, 17, 17), 2, cv2.LINE_AA)

    # -------------------------- MAIN LOOP --------------------------
    def run(self):
        # Open Camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("\nError: Could not access webcam (index 0).")
            print("Please check that camera permissions are enabled for your terminal / IDE in macOS System Settings.")
            return

        window_name = "One-More: Strict AI Pushup Coach"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.on_mouse)

        pose = mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        print("One-More Pushup Coach started.")
        print("Click [START] or press 'S' to begin 10-second countdown.")
        print("Click [STOP] or press 'Space' to finish.")
        print("Press 'Q' to quit.\n")

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            now = time.time()
            h, w = frame.shape[:2]

            # Clear main canvas to warm paper white
            canvas[:] = (245, 249, 250)

            # Process MediaPipe Pose
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            valid = False
            sh = hip = ank = None

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark

                # Determine best side
                if self.side is None or self.state == "WAIT":
                    new_side = choose_best_side(lm)
                    if new_side != self.side:
                        self.side = new_side
                        self.elbow_hist.clear()
                        self.body_hist.clear()

                ids = SIDES[self.side]
                vis = min(lm[i].visibility for i in ids)

                if vis >= MIN_VISIBILITY:
                    # Joint coordinates in image pixels
                    sh, el, wr, hip, ank = [np.array([lm[i].x * w, lm[i].y * h]) for i in ids]
                    self.elbow_hist.append(calc_angle(sh, el, wr))
                    self.body_hist.append(calc_angle(sh, hip, ank))
                    self.elbow_angle = float(np.mean(self.elbow_hist))
                    self.body_angle = float(np.mean(self.body_hist))

                    valid = is_plank_geometry_ok(sh, wr, ank)
                    self.status_msg = "In pushup position" if valid else "Not horizontal plank"
                else:
                    self.status_msg = "Move back — need full body view"

                is_good_form = valid and (self.body_angle >= BODY_MIN if self.body_angle else False)
                self.draw_skeleton(frame, lm, w, h, is_good_form)
            else:
                self.status_msg = "No person detected"

            # Execute rep state machine if in active workout
            if self.mode == AppMode.WORKOUT:
                self.update_workout_logic(valid, sh, hip, ank, self.elbow_angle, self.body_angle, now)

            # Render Overlays on Video Feed
            self.render_viewport_overlays(frame, now)

            # Fit Video Feed into Viewport (860 x 680)
            scale = min(VIEWPORT_W / w, VIEWPORT_H / h)
            nw, nh = int(w * scale), int(h * scale)
            resized_video = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

            vx = VIEWPORT_X + (VIEWPORT_W - nw) // 2
            vy = VIEWPORT_Y + (VIEWPORT_H - nh) // 2

            # Card background for viewport
            draw_card(canvas, VIEWPORT_X, VIEWPORT_Y, VIEWPORT_W, VIEWPORT_H, bg_color=(17, 17, 17), border_color=(17, 17, 17))
            canvas[vy:vy + nh, vx:vx + nw] = resized_video

            # Render Dashboard Sidebar
            self.render_sidebar(canvas, now)

            # Display composite window
            cv2.imshow(window_name, canvas)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                break
            elif key in (ord('s'), ord('S')):    # S to Start
                if self.mode in (AppMode.IDLE, AppMode.STOPPED):
                    self.start_countdown()
            elif key in (ord(' '), ord('x'), ord('X')):  # Space or X to Stop / Start
                if self.mode in (AppMode.COUNTDOWN, AppMode.WORKOUT):
                    self.stop_workout()
                elif self.mode in (AppMode.IDLE, AppMode.STOPPED):
                    self.start_countdown()

        pose.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = PushupApp()
    app.run()
