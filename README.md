# One-More: AI Strict Pushup Counter & Form Coach

An AI-powered pushup counter built with OpenCV and MediaPipe Pose that counts pushups **only when performed with strict, correct form**.

---

## Features

- **Interactive Dark Dashboard UI**:
  - Widescreen 1280x720 layout with dedicated video viewport and stats dashboard.
  - Interactive **[START]** and **[STOP]** buttons (mouse-clickable or via keyboard shortcuts).
- **10-Second Preparation Countdown**:
  - Clicking **[START]** triggers an animated 10-second countdown with visual instructions and audio announcements, giving you time to get into position.
- **Precision Workout Timer**:
  - Automatically starts ticking (`MM:SS.t`) exactly 10 seconds after clicking Start.
  - Stops immediately when you click **[STOP]**.
- **Strict Form Validation**:
  - **1-Second Plank Hold**: Requires holding a straight plank still for 1 second before counting begins (ignores walking or arm movement).
  - **Full Depth**: Elbow angle must reach $\le 90^\circ$ at the bottom.
  - **Full Lockout**: Elbow angle must return to $\ge 160^\circ$ at the top.
  - **Strict Spine Alignment**: Shoulder-hip-ankle line must stay $\ge 160^\circ$ throughout the entire rep (rejects sagging hips or piking).
  - **Cadence & Bouncing Rejection**: Reps must take at least 0.8 seconds (rejects rapid bouncing or jitter).
  - **Real-Time Guidance**: Displays and announces specific form corrections ("Go lower", "Hips too high", "Hips sagging").
- **Session Summary**:
  - Displays session duration, strict reps, no-point reps, form accuracy percentage, and reps-per-minute cadence upon completion.

---

## Setup & Running

You have two ways to run the UI:

### Option A: Modern Web Dashboard (Browser)
1. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```
2. **Launch the web server**:
   ```bash
   python app_web.py
   ```
3. Open your browser at: **[http://localhost:8000](http://localhost:8000)**

---

### Option B: Native Desktop Window (OpenCV)
1. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```
2. **Launch the desktop window**:
   ```bash
   python pushup_strict.py
   ```

---

## Controls

| Action | Mouse | Keyboard |
|---|---|---|
| **Start (10s Prep Countdown)** | Click **[START]** button | Press `S` |
| **Stop Workout & View Summary** | Click **[STOP]** button | Press `Space` or `X` |
| **Quit Application** | Close window | Press `Q` or `ESC` |

---

## Camera Setup Tips

- **Camera Placement**: Place your phone/camera directly at your **side view** (perpendicular, not at an angle).
- **Height**: Roughly chest/floor level, about 2–3 meters away.
- **Framing**: Ensure your head, hands, hips, and feet are fully visible in the camera view.

