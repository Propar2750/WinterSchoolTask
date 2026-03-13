# Bonus Task: Gesture-Controlled Tetris — Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Tetris Game Engine (`TetrisGame`)](#tetris-game-engine)
4. [Rendering (`draw_game`, `draw_help`)](#rendering)
5. [Gesture Controller (`GestureController`)](#gesture-controller)
6. [Main Game Loop](#main-game-loop)
7. [Library Functions Used](#library-functions-used)
8. [Dependencies](#dependencies)

---

## Overview

This project implements a fully playable Tetris game rendered with OpenCV, controllable via both keyboard and real-time hand gestures detected through a webcam using MediaPipe's Hand Landmarker model. The gesture system uses pinch detection (thumb touching index/middle finger) for lateral movement, a thumbs-up for rotation, and a closed fist for hard drop.

---

## Architecture

The program is structured into four main components:

```
bonustask.py
├── Constants & Config       (grid size, colors, tetromino shapes)
├── TetrisGame class         (game state, movement, collision, scoring)
├── draw_game / draw_help    (OpenCV rendering functions)
├── GestureController class  (webcam + MediaPipe hand tracking + gesture-to-action mapping)
└── main()                   (game loop tying everything together)
```

---

## Tetris Game Engine

### Class: `TetrisGame`

Manages the entire game state: the grid, the falling piece, scoring, and level progression.

### Data Structures

- **`self.grid`**: A 2D NumPy array of shape `(ROWS=20, COLS=10)` filled with 0 (empty) or 1 (occupied).
- **`self.grid_colors`**: A dictionary mapping `(row, col)` tuples to piece-type strings (e.g., `"T"`, `"L"`) for coloring locked cells.
- **`SHAPES`**: A dictionary where each key is a piece name and each value is a list of rotations. Each rotation is a list of `(row_offset, col_offset)` tuples relative to the piece's anchor.

### Pseudocode: Spawning a Piece

```
FUNCTION spawn():
    current_piece ← next_piece
    next_piece ← random choice from piece names
    rotation ← 0
    position ← (row=0, col=center of grid)
    IF the piece immediately collides at spawn position:
        game_over ← true
```

### Pseudocode: Collision Detection

```
FUNCTION collides(row, col, rotation):
    offsets ← SHAPES[current_piece][rotation mod num_rotations]
    FOR EACH (dr, dc) in offsets:
        r ← row + dr
        c ← col + dc
        IF r or c is out of grid bounds:
            RETURN true
        IF grid[r][c] is occupied:
            RETURN true
    RETURN false
```

### Pseudocode: Movement

```
FUNCTION move(dr, dc):
    IF game is over: RETURN
    new_row ← piece_row + dr
    new_col ← piece_col + dc
    IF NOT collides(new_row, new_col, current_rotation):
        piece_row ← new_row
        piece_col ← new_col
        RETURN true
    RETURN false
```

### Pseudocode: Rotation with Wall Kicks

```
FUNCTION rotate():
    IF game is over: RETURN
    new_rotation ← (current_rotation + 1) mod num_rotations
    FOR kick IN [0, -1, +1]:        // try center, then nudge left, then right
        IF NOT collides(piece_row, piece_col + kick, new_rotation):
            current_rotation ← new_rotation
            piece_col ← piece_col + kick
            RETURN
```

**Wall kicks** handle the case where a rotation would push the piece into a wall. The algorithm tries the rotation at the current column first, then shifts left by 1, then right by 1. The first non-colliding position wins.

### Pseudocode: Hard Drop

```
FUNCTION hard_drop():
    IF game is over: RETURN
    WHILE move(down by 1) succeeds:
        continue            // keep dropping
    lock()                  // piece can't go further, lock it
```

### Pseudocode: Locking and Line Clearing

```
FUNCTION lock():
    FOR EACH cell (r, c) of the current piece:
        grid[r][c] ← 1 (occupied)
        grid_colors[(r,c)] ← piece type
    clear_lines()
    spawn()                 // bring in the next piece

FUNCTION clear_lines():
    full_rows ← list of rows where every cell is occupied
    IF no full rows: RETURN

    // Scoring: 1 line=100, 2=300, 3=500, 4=800 (multiplied by level)
    score ← score + points[num_full_rows] * level
    lines_cleared ← lines_cleared + num_full_rows
    level ← 1 + lines_cleared / 10   (integer division)

    // Compact the grid: copy non-full rows downward
    new_grid ← empty grid
    dest ← bottom row
    FOR src FROM bottom row TO top row:
        IF src is NOT a full row:
            copy grid[src] → new_grid[dest]
            copy corresponding colors
            dest ← dest - 1
    grid ← new_grid
```

### Pseudocode: Gravity / Drop Speed

```
FUNCTION get_drop_speed():
    RETURN max(0.05, 0.5 - (level - 1) * 0.04)
    // Level 1: 0.50s, Level 2: 0.46s, ..., Level 12+: 0.05s (fastest)
```

---

## Rendering

### `draw_game(game)`

Renders the full game frame as an OpenCV image (NumPy BGR array).

```
FUNCTION draw_game(game):
    img ← blank image of size (WIN_H × WIN_W), filled with BG_COLOR

    // 1. Draw locked cells
    FOR EACH cell (r, c) in the grid:
        IF occupied:
            draw filled rectangle at pixel position with the piece's color

    // 2. Draw ghost piece (drop preview)
    ghost_row ← current piece row
    WHILE piece doesn't collide at (ghost_row + 1):
        ghost_row ← ghost_row + 1
    draw piece cells at ghost_row with dimmed color (1/3 brightness)

    // 3. Draw actual falling piece
    draw piece cells at current position with full color

    // 4. Draw grid lines (horizontal and vertical)

    // 5. Draw sidebar: title, score, lines, level, next piece preview, controls hint

    // 6. If game over: dark overlay + "GAME OVER" text + score + restart hint

    RETURN img
```

### `draw_help(base_img)`

Draws a translucent help overlay on top of the game frame.

```
FUNCTION draw_help(base_img):
    overlay ← copy of base_img
    fill overlay with black
    img ← alpha-blend base_img (15%) with overlay (85%)

    draw title "HELP"
    draw separator line

    draw section "Keyboard Controls" with key mappings
    draw section "Gesture Controls" with gesture descriptions
    draw section "Gesture Tips" with usage advice

    draw "Press H to close" at bottom

    RETURN img
```

---

## Gesture Controller

### Class: `GestureController`

Handles webcam capture, MediaPipe hand landmark detection, gesture classification, and mapping gestures to game actions.

### MediaPipe Hand Landmarker — How It Works

MediaPipe's Hand Landmarker is a pre-trained deep learning model that detects hands in images and returns 21 3D landmark points per hand. The landmarks correspond to anatomical joints:

```
Landmark indices:
  0  = Wrist
  1-4   = Thumb (CMC, MCP, IP, TIP)
  5-8   = Index finger (MCP, PIP, DIP, TIP)
  9-12  = Middle finger (MCP, PIP, DIP, TIP)
  13-16 = Ring finger (MCP, PIP, DIP, TIP)
  17-20 = Pinky (MCP, PIP, DIP, TIP)
```

Each landmark has normalized `(x, y, z)` coordinates where `x` and `y` are in `[0, 1]` relative to image width/height, and `z` represents depth relative to the wrist.

The model runs in `VIDEO` mode, which uses temporal consistency (tracking between frames) for smoother results than processing each frame independently.

### Pseudocode: Initialization

```
FUNCTION __init__():
    cap ← None                    // webcam capture (opened on demand)
    landmarker ← None             // MediaPipe model (loaded on demand)
    last_action_time ← {}         // tracks last trigger time per action
    cooldowns ← {
        move_left:  0.25s,
        move_right: 0.25s,
        rotate:     0.60s,
        hard_drop:  1.00s
    }
    pinch_threshold ← 0.07       // normalized distance to count as "touching"
    pinch_fired ← false           // one-shot flag, prevents repeat triggers
```

### Pseudocode: Cooldown Check

The cooldown system prevents rapid-fire triggering of the same action.

```
FUNCTION cooled_down(action):
    now ← current time
    last ← last_action_time[action] or 0
    IF (now - last) >= cooldowns[action]:
        last_action_time[action] ← now
        RETURN true
    RETURN false
```

### Pseudocode: Finger-Up Detection

Determines which fingers are extended (up) vs curled (down).

```
FUNCTION fingers_up(landmarks):
    fingers ← []

    // Thumb: compare tip.x vs IP joint.x
    // (works for right hand facing a mirrored camera)
    IF thumb_tip.x < thumb_ip.x:
        fingers[0] ← true (thumb is out)

    // Index, Middle, Ring, Pinky: compare tip.y vs PIP joint.y
    // In image coordinates, y=0 is top, so tip.y < pip.y means finger points up
    FOR EACH finger IN [index, middle, ring, pinky]:
        IF finger_tip.y < finger_pip.y:
            fingers[i] ← true (finger is extended)

    RETURN fingers    // list of 5 booleans
```

### Pseudocode: Pinch Detection (Core Gesture Logic)

```
FUNCTION classify_gesture(landmarks, game):
    fingers ← fingers_up(landmarks)

    // ── Gesture 1: Closed Fist (all fingers down) ──
    IF no finger is up:
        action ← "Hard Drop"
        IF cooled_down("hard_drop"):
            game.hard_drop()

    // ── Gesture 2: Thumb Only (thumb up, all others down) ──
    ELSE IF thumb is up AND no other finger is up:
        action ← "Rotate"
        IF cooled_down("rotate"):
            game.rotate()

    // ── Gesture 3: Pinch Detection (any other hand pose) ──
    ELSE:
        thumb_pos ← landmarks[4]    // thumb tip
        index_pos ← landmarks[8]    // index tip
        middle_pos ← landmarks[12]  // middle tip

        // Euclidean distance in normalized coordinates
        d_index  ← sqrt((thumb.x - index.x)^2 + (thumb.y - index.y)^2)
        d_middle ← sqrt((thumb.x - middle.x)^2 + (thumb.y - middle.y)^2)

        IF pinch has NOT been fired yet:
            IF d_index < pinch_threshold (0.07):
                action ← "Move Left"
                IF cooled_down("move_left"):
                    game.move(0, -1)
                pinch_fired ← true       // lock out until reset

            ELSE IF d_middle < pinch_threshold (0.07):
                action ← "Move Right"
                IF cooled_down("move_right"):
                    game.move(0, 1)
                pinch_fired ← true

            ELSE:
                action ← "Open hand (ready)"

        ELSE:  // pinch was already fired, waiting for reset
            // Reset when BOTH fingers are far enough from thumb
            // Hysteresis: require 1.2x threshold to reset (prevents flickering)
            IF d_index > threshold * 1.2 AND d_middle > threshold * 1.2:
                pinch_fired ← false
                action ← "Open hand (ready)"
            ELSE:
                action ← "Waiting for reset"

    // If no hand detected at all:
    IF no hand in frame:
        pinch_fired ← false     // auto-reset
```

### Why Hysteresis for Reset?

Without hysteresis, the system would rapidly toggle between "pinched" and "not pinched" when the finger distance hovers near the threshold. By requiring a slightly larger distance (1.2x) to reset than to trigger (1.0x), we create a dead zone that prevents flickering:

```
  Trigger zone:   distance < 0.07   → fire action, set pinch_fired = true
  Dead zone:      0.07 ≤ distance ≤ 0.084   → no change
  Reset zone:     distance > 0.084  → reset pinch_fired = false
```

### Pseudocode: Full Frame Processing Pipeline

```
FUNCTION process(game):
    // 1. Capture frame from webcam
    ret, frame ← cap.read()
    IF not ret: RETURN None

    // 2. Mirror the frame (so hand movements match screen direction)
    frame ← flip(frame, horizontally)

    // 3. Convert BGR → RGB and wrap as MediaPipe Image
    rgb_data ← cvtColor(frame, BGR2RGB)
    mp_image ← MediaPipe.Image(format=SRGB, data=rgb_data)

    // 4. Run hand landmark detection
    frame_timestamp ← frame_timestamp + 33ms   // simulate ~30fps
    result ← landmarker.detect_for_video(mp_image, frame_timestamp)

    // 5. If hand detected: draw skeleton, classify gesture, apply action
    IF result has landmarks:
        draw_landmarks(frame, landmarks)
        classify_gesture(landmarks, game)
    ELSE:
        pinch_fired ← false   // reset state when hand leaves

    // 6. Draw HUD text on webcam frame
    draw action label at top
    draw control legend at bottom

    RETURN frame
```

### Pseudocode: Drawing Hand Skeleton

```
FUNCTION draw_landmarks(frame, landmarks):
    h, w ← frame dimensions

    // Convert normalized (0-1) coordinates to pixel coordinates
    points ← []
    FOR EACH landmark:
        px ← int(landmark.x * w)
        py ← int(landmark.y * h)
        points.append((px, py))

    // Draw bones (connections between joints)
    FOR EACH (i, j) in CONNECTIONS:
        draw line from points[i] to points[j], green, thickness=2

    // Draw joints
    FOR EACH point:
        draw filled circle at point, red, radius=4
```

---

## Main Game Loop

```
FUNCTION main():
    game ← new TetrisGame()
    gesture ← new GestureController()
    gesture_mode ← false
    show_help ← false
    last_drop ← current time

    create OpenCV window "Tetris"

    LOOP FOREVER:
        now ← current time

        // 1. Process gestures (if enabled)
        IF gesture_mode:
            cam_frame ← gesture.process(game)
            IF cam_frame exists:
                show cam_frame in "Webcam" window

        // 2. Apply gravity
        IF game not over AND (now - last_drop) >= drop_speed:
            game.step()       // move piece down 1 row
            last_drop ← now

        // 3. Render game
        frame ← draw_game(game)
        IF gesture_mode: draw "[GESTURE ON]" indicator
        IF show_help: frame ← draw_help(frame)
        display frame

        // 4. Handle keyboard input (polled every 30ms)
        key ← waitKey(30ms)
        SWITCH key:
            'q' → break loop
            'r' → game.reset()
            'h' → toggle show_help
            'g' → toggle gesture_mode (start/stop webcam)
            arrows/WASD → move/rotate piece
            space → hard_drop

    // Cleanup
    gesture.release()
    destroy all OpenCV windows
```

---

## Library Functions Used

### OpenCV (`cv2`)

| Function | Purpose | How It Works |
|---|---|---|
| `cv2.VideoCapture(0)` | Opens the default webcam (device index 0) and returns a capture object. Frames are read from it via `.read()`. |
| `cap.read()` | Reads one frame from the webcam. Returns `(success_bool, frame_ndarray)`. The frame is a BGR image as a NumPy array of shape `(H, W, 3)`. |
| `cv2.flip(frame, 1)` | Flips the image horizontally (flipCode=1). This mirrors the webcam feed so the user's left hand appears on the left side of the screen. |
| `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` | Converts pixel color space from BGR (OpenCV default) to RGB (MediaPipe requirement). Rearranges the 3 channels of each pixel. |
| `cv2.rectangle(img, pt1, pt2, color, thickness)` | Draws a rectangle. `pt1`/`pt2` are opposite corners. `thickness=-1` fills the rectangle. Used for drawing Tetris cells. |
| `cv2.line(img, pt1, pt2, color, thickness)` | Draws a straight line between two points. Used for grid lines and hand skeleton bones. |
| `cv2.circle(img, center, radius, color, thickness)` | Draws a circle. `thickness=-1` fills it. Used for hand landmark joint dots. |
| `cv2.putText(img, text, org, font, scale, color, thickness)` | Renders text onto an image at position `org` (bottom-left corner of text). `font` selects the typeface, `scale` controls size. |
| `cv2.addWeighted(img1, alpha, img2, beta, gamma)` | Computes pixel-wise: `output = img1*alpha + img2*beta + gamma`. Used for translucent overlays (help screen, game-over dimming). |
| `cv2.namedWindow(name, flags)` | Creates a named window. `WINDOW_AUTOSIZE` makes it resize to fit the image exactly. |
| `cv2.imshow(name, img)` | Displays an image in the named window. The window updates when this is called. |
| `cv2.waitKey(ms)` | Waits up to `ms` milliseconds for a keypress. Returns the key code (or -1 if none). Also pumps the GUI event loop — without it, windows won't update. The `& 0xFF` mask extracts the last byte for cross-platform compatibility. |
| `cv2.destroyWindow(name)` / `cv2.destroyAllWindows()` | Closes one or all OpenCV windows and frees their resources. |
| `cap.release()` | Releases the webcam device so other applications can use it. |

### NumPy (`np`)

| Function | Purpose | How It Works |
|---|---|---|
| `np.zeros((rows, cols), dtype=int)` | Creates a 2D array filled with zeros. Used for the game grid where 0 = empty cell. |
| `np.full((H, W, 3), color, dtype=np.uint8)` | Creates an array filled with a constant value. Used to create a blank BGR image with the background color. |
| `np.all(array != 0)` | Returns `True` if every element in the array is non-zero. Used to check if a grid row is completely filled (for line clearing). |

### MediaPipe (`mediapipe`)

| Function / Class | Purpose | How It Works |
|---|---|---|
| `mp.tasks.BaseOptions(model_asset_path=...)` | Configuration object specifying the path to the `.task` model file (a TFLite model bundle). |
| `HandLandmarkerOptions(...)` | Configuration for the hand landmarker: `running_mode=VIDEO` enables temporal tracking between frames; `num_hands=1` limits detection to one hand; confidence thresholds control sensitivity. |
| `HandLandmarker.create_from_options(options)` | Factory method that loads the model into memory and returns a landmarker object ready for inference. |
| `mp.Image(image_format=SRGB, data=rgb_array)` | Wraps a NumPy RGB array as a MediaPipe Image object that the landmarker can process. |
| `landmarker.detect_for_video(mp_image, timestamp_ms)` | Runs hand detection on one video frame. The timestamp must be monotonically increasing (we use +33ms per frame ≈ 30fps). Returns a result object containing `hand_landmarks` — a list of detected hands, each being a list of 21 `NormalizedLandmark` objects with `.x`, `.y`, `.z` fields in `[0, 1]`. |
| `landmarker.close()` | Releases the model and its resources. |

### Python Standard Library

| Function | Purpose |
|---|---|
| `time.time()` | Returns current time in seconds (float). Used for cooldown timing and gravity intervals. |
| `random.choice(list)` | Returns a random element from the list. Used to select the next Tetris piece. |
| `os.path.dirname/abspath/exists/join` | Path manipulation: `abspath(__file__)` gets the script's absolute path; `dirname` gets its directory; `join` constructs paths; `exists` checks if the model file is already downloaded. |
| `urllib.request.urlretrieve(url, path)` | Downloads a file from a URL and saves it to a local path. Used for auto-downloading the MediaPipe model on first run. |
| `math via **` operator | `((a-b)**2 + (c-d)**2) ** 0.5` computes Euclidean distance without importing `math`. |

---

## Dependencies

```bash
pip install opencv-python numpy mediapipe
```

- **Python** 3.10+
- **opencv-python** — rendering, webcam capture, keyboard input
- **numpy** — game grid, image arrays
- **mediapipe** — hand landmark detection

The `hand_landmarker.task` model file (~10 MB) is auto-downloaded on first run if not present.
