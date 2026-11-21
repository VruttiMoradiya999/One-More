"""
Strict pushup counter (high accuracy version).

Install:  pip install mediapipe==0.10.14 opencv-python numpy
Run:      python pushup_strict.py        (press 'q' to quit)

Camera setup (IMPORTANT for accuracy):
  - Exactly SIDE view (camera at your side, not diagonal)
  - Camera low, about chest/floor height, 2-3 metres away
  - Hands, head and feet all visible. Good lighting. Plain background helps.

How a rep is counted:
  1. You get into plank position and hold still for 1 second -> "READY".
     Nothing is counted before this, so walking around / moving arms is ignored.
  2. Elbow goes down to <= 90 degrees (held for a few frames).
  3. Elbow comes back up to >= 160 degrees (held for a few frames).
  4. Body (shoulder-hip-ankle) stayed >= 160 degrees the whole rep.
  5. Rep took at least 0.8 s (stops jitter / bouncing from counting).
  Otherwise: no point, and the reason is shown.
"""

import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# ============================ SETTINGS (tweak here) ==========================
COUNTDOWN_SEC = 2

ELBOW_DOWN = 90        # elbow must reach this or lower (full depth)
ELBOW_UP = 160         # elbow must come back to this or higher (lockout)
ELBOW_START = 150      # below this a rep is considered started
BODY_MIN = 160         # shoulder-hip-ankle angle must stay at/above this

MAX_TILT_DEG = 30      # body must be within this many degrees of horizontal
HANDS_UNDER_SHOULDER = 0.35   # wrist x-distance from shoulder, as fraction of body length
MIN_VISIBILITY = 0.6   # all 5 joints must be seen at least this well

READY_HOLD_SEC = 1.0   # hold plank this long before counting starts
LOST_POSITION_SEC = 0.6   # leaving position this long cancels the rep
CONFIRM_FRAMES = 3     # a threshold must hold this many frames in a row
MIN_REP_SEC = 0.8      # faster than this = jitter/bounce, not counted
MAX_REP_SEC = 15.0     # slower than this = rep cancelled
SMOOTH = 5             # frames averaged to remove landmark jitter

FRAME_W, FRAME_H = 640, 480
MODEL_COMPLEXITY = 1   # 0 = fastest, 1 = more accurate (recommended), 2 = heaviest
# =============================================================================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
P = mp_pose.PoseLandmark

SIDES = {
    "left": [P.LEFT_SHOULDER, P.LEFT_ELBOW, P.LEFT_WRIST, P.LEFT_HIP, P.LEFT_ANKLE],
    "right": [P.RIGHT_SHOULDER, P.RIGHT_ELBOW, P.RIGHT_WRIST, P.RIGHT_HIP, P.RIGHT_ANKLE],
}


def angle(a, b, c):
    """Angle at point b (degrees) formed by a-b-c. Points are in PIXELS."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def best_side(lm):
    """Choose the body side the camera sees most clearly."""
    scores = {s: min(lm[i].visibility for i in ids) for s, ids in SIDES.items()}
    return max(scores, key=scores.get)


def plank_geometry_ok(sh, wr, ank):
    """Is the person actually lying horizontally with hands under shoulders?"""
    dx, dy = abs(sh[0] - ank[0]), abs(sh[1] - ank[1])
    tilt = np.degrees(np.arctan2(dy, dx + 1e-9))
    body_len = np.hypot(dx, dy) + 1e-9
    horizontal = tilt <= MAX_TILT_DEG
    hands_under = abs(wr[0] - sh[0]) <= HANDS_UNDER_SHOULDER * body_len
    return horizontal and hands_under


def hip_problem(sh, hip, ank):
    """Say whether hips are too high (pike) or too low (sag)."""
    if abs(ank[0] - sh[0]) < 1e-6:
        return "keep body straight"
    t = (hip[0] - sh[0]) / (ank[0] - sh[0])
    line_y = sh[1] + t * (ank[1] - sh[1])
    return "hips too high" if hip[1] < line_y else "hips too low"


def put(frame, text, pos, color=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    pose = mp_pose.Pose(model_complexity=MODEL_COMPLEXITY,
                        smooth_landmarks=True,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6)

    start_time = time.time()
    score, bad_reps = 0, 0

    state = "WAIT"            # WAIT -> UP -> LOWERING -> DOWN -> UP ...
    side = None
    elbow_hist = deque(maxlen=SMOOTH)
    body_hist = deque(maxlen=SMOOTH)

    hold_start = None         # when the plank hold began
    lost_since = None         # when the person left a valid position
    rep_start = 0.0
    body_ok = True
    problem = ""
    down_cnt = up_cnt = body_bad_cnt = 0

    feedback, feedback_time = "", 0.0
    status = "Can't see you"
    elbow = body = None

    def reset_rep_counters():
        nonlocal down_cnt, up_cnt, body_bad_cnt
        down_cnt = up_cnt = body_bad_cnt = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        now = time.time()

        # ------------------------- countdown -------------------------------
        elapsed = now - start_time
        if elapsed < COUNTDOWN_SEC:
            put(frame, "Place phone & get into position", (20, 60), (0, 255, 255), 0.8)
            put(frame, f"Starting in {int(COUNTDOWN_SEC - elapsed) + 1}", (20, 110), (0, 255, 255), 1.1)
            cv2.imshow("Pushup Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # ------------------------- pose detection --------------------------
        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        valid = False
        sh = hip = ank = None

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Only switch body side while waiting; lock it once ready
            if side is None or state == "WAIT":
                new_side = best_side(lm)
                if new_side != side:
                    side = new_side
                    elbow_hist.clear()
                    body_hist.clear()

            ids = SIDES[side]
            vis = min(lm[i].visibility for i in ids)

            if vis >= MIN_VISIBILITY:
                # Convert to PIXELS so angles are not distorted by aspect ratio
                sh, el, wr, hip, ank = [np.array([lm[i].x * w, lm[i].y * h]) for i in ids]
                elbow_hist.append(angle(sh, el, wr))
                body_hist.append(angle(sh, hip, ank))
                elbow = float(np.mean(elbow_hist))
                body = float(np.mean(body_hist))
                valid = plank_geometry_ok(sh, wr, ank)
                status = "In position" if valid else "Not in pushup position"
            else:
                status = "Can't see full body - move camera back"
        else:
            status = "No person detected"

        # ------------------------- state machine ---------------------------
        if not valid:
            hold_start = None
            if lost_since is None:
                lost_since = now
            if state != "WAIT" and now - lost_since > LOST_POSITION_SEC:
                if state in ("LOWERING", "DOWN"):
                    feedback, feedback_time = "Rep cancelled: left position", now
                state = "WAIT"
                reset_rep_counters()
                elbow_hist.clear()
                body_hist.clear()
        else:
            lost_since = None

            if state == "WAIT":
                # Must hold a straight plank, arms extended, before counting starts
                if elbow >= ELBOW_UP and body >= BODY_MIN:
                    if hold_start is None:
                        hold_start = now
                    if now - hold_start >= READY_HOLD_SEC:
                        state = "UP"
                        feedback, feedback_time = "READY! Start your pushups", now
                else:
                    hold_start = None

            elif state == "UP":
                if elbow < ELBOW_START:
                    state = "LOWERING"
                    rep_start = now
                    body_ok, problem = True, ""
                    reset_rep_counters()

            elif state in ("LOWERING", "DOWN"):
                # --- body straightness (must stay straight the whole rep) ---
                if body < BODY_MIN:
                    body_bad_cnt += 1
                else:
                    body_bad_cnt = 0
                if body_bad_cnt >= CONFIRM_FRAMES and body_ok:
                    body_ok = False
                    problem = hip_problem(sh, hip, ank)

                # --- too slow ---
                if now - rep_start > MAX_REP_SEC:
                    feedback, feedback_time = "Rep cancelled: too slow", now
                    state = "UP"
                    reset_rep_counters()

                elif state == "LOWERING":
                    down_cnt = down_cnt + 1 if elbow <= ELBOW_DOWN else 0
                    if down_cnt >= CONFIRM_FRAMES:
                        state = "DOWN"
                        up_cnt = 0
                    else:
                        up_cnt = up_cnt + 1 if elbow >= ELBOW_UP else 0
                        if up_cnt >= CONFIRM_FRAMES:
                            bad_reps += 1
                            feedback, feedback_time = "No point: go lower", now
                            state = "UP"
                            reset_rep_counters()

                elif state == "DOWN":
                    up_cnt = up_cnt + 1 if elbow >= ELBOW_UP else 0
                    if up_cnt >= CONFIRM_FRAMES:
                        problems = []
                        if not body_ok:
                            problems.append(problem or "keep body straight")
                        if now - rep_start < MIN_REP_SEC:
                            problems.append("too fast")
                        if problems:
                            bad_reps += 1
                            feedback = "No point: " + ", ".join(problems)
                        else:
                            score += 1
                            feedback = "GOOD REP!"
                        feedback_time = now
                        state = "UP"
                        reset_rep_counters()

        # ------------------------- on-screen info --------------------------
        put(frame, f"Score: {score}", (15, 40), (0, 255, 0), 1.1, 3)
        put(frame, f"No-point reps: {bad_reps}", (15, 75), (80, 80, 255), 0.6)

        if state == "WAIT":
            if hold_start is not None:
                held = min(now - hold_start, READY_HOLD_SEC)
                msg = f"Hold still... {held:.1f}/{READY_HOLD_SEC:.0f}s"
            else:
                msg = "Get into plank position (side view)"
            put(frame, msg, (15, 110), (0, 255, 255), 0.65)
        else:
            put(frame, {"UP": "Ready - go down", "LOWERING": "Going down...",
                        "DOWN": "Now push up!"}[state], (15, 110), (0, 255, 255), 0.65)

        if feedback and now - feedback_time < 2.0:
            color = (0, 255, 0) if feedback in ("GOOD REP!",) or feedback.startswith("READY") \
                else (0, 0, 255)
            put(frame, feedback, (15, 150), color, 0.8, 2)

        if elbow is not None and body is not None:
            put(frame, f"Elbow:{int(elbow)}  Body:{int(body)}  [{status}]",
                (15, h - 15), (255, 255, 0), 0.5, 1)

        cv2.imshow("Pushup Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()