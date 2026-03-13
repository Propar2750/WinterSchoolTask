# Bonus Task: Gesture-Controlled Tetris
# OpenCV display + keyboard controls + MediaPipe hand gesture controls
# Press 'g' to toggle gesture mode, 'h' for help, 'q' to quit, 'r' to restart

import cv2
import numpy as np
import os
import random
import time

# ── Grid & Display Config ──────────────────────────────────────────
COLS, ROWS = 10, 20
CELL = 30
SIDEBAR_W = 180
GAME_W = COLS * CELL
GAME_H = ROWS * CELL
WIN_W = GAME_W + SIDEBAR_W
WIN_H = GAME_H

# ── Colors (BGR) ───────────────────────────────────────────────────
BG_COLOR = (30, 30, 30)
GRID_COLOR = (50, 50, 50)
TEXT_COLOR = (220, 220, 220)
PIECE_COLORS = {
    "I": (200, 180, 50),   # cyan-ish
    "O": (50, 210, 210),   # yellow
    "T": (180, 50, 180),   # purple
    "L": (50, 130, 220),   # orange
    "S": (50, 200, 50),    # green
}

# ── Tetromino Shapes (row, col offsets for each rotation) ──────────
# Each shape has a list of rotations; each rotation is a list of (row, col) offsets
SHAPES = {
    "I": [
        [(0, -1), (0, 0), (0, 1), (0, 2)],
        [(-1, 0), (0, 0), (1, 0), (2, 0)],
    ],
    "O": [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    "T": [
        [(0, -1), (0, 0), (0, 1), (1, 0)],
        [(-1, 0), (0, 0), (1, 0), (0, 1)],
        [(-1, 0), (0, -1), (0, 0), (0, 1)],
        [(-1, 0), (0, 0), (1, 0), (0, -1)],
    ],
    "L": [
        [(0, -1), (0, 0), (0, 1), (1, 1)],
        [(-1, 0), (0, 0), (1, 0), (1, -1)],
        [(-1, -1), (0, -1), (0, 0), (0, 1)],
        [(-1, 0), (-1, 1), (0, 0), (1, 0)],
    ],
    "S": [
        [(0, 0), (0, 1), (1, -1), (1, 0)],
        [(-1, 0), (0, 0), (0, 1), (1, 1)],
    ],
}

SHAPE_NAMES = list(SHAPES.keys())


class TetrisGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = np.zeros((ROWS, COLS), dtype=int)
        self.grid_colors = {}  # (row, col) -> color name
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.game_over = False
        self.next_piece = random.choice(SHAPE_NAMES)
        self.spawn()

    def spawn(self):
        self.piece = self.next_piece
        self.next_piece = random.choice(SHAPE_NAMES)
        self.rotation = 0
        self.piece_row = 0
        self.piece_col = COLS // 2
        if self._collides(self.piece_row, self.piece_col, self.rotation):
            self.game_over = True

    def _get_cells(self, row, col, rotation):
        offsets = SHAPES[self.piece][rotation % len(SHAPES[self.piece])]
        return [(row + dr, col + dc) for dr, dc in offsets]

    def _collides(self, row, col, rotation):
        for r, c in self._get_cells(row, col, rotation):
            if r < 0 or r >= ROWS or c < 0 or c >= COLS:
                return True
            if self.grid[r, c] != 0:
                return True
        return False

    def move(self, dr, dc):
        if self.game_over:
            return
        new_r = self.piece_row + dr
        new_c = self.piece_col + dc
        if not self._collides(new_r, new_c, self.rotation):
            self.piece_row = new_r
            self.piece_col = new_c
            return True
        return False

    def rotate(self):
        if self.game_over:
            return
        new_rot = (self.rotation + 1) % len(SHAPES[self.piece])
        # Try basic rotation, then wall kicks (shift left/right by 1)
        for kick in [0, -1, 1]:
            if not self._collides(self.piece_row, self.piece_col + kick, new_rot):
                self.rotation = new_rot
                self.piece_col += kick
                return

    def hard_drop(self):
        if self.game_over:
            return
        while self.move(1, 0):
            pass
        self._lock()

    def step(self):
        """Gravity: move piece down one row. Lock if can't."""
        if self.game_over:
            return
        if not self.move(1, 0):
            self._lock()

    def _lock(self):
        for r, c in self._get_cells(self.piece_row, self.piece_col, self.rotation):
            if 0 <= r < ROWS and 0 <= c < COLS:
                self.grid[r, c] = 1
                self.grid_colors[(r, c)] = self.piece
        self._clear_lines()
        self.spawn()

    def _clear_lines(self):
        full_rows = [r for r in range(ROWS) if np.all(self.grid[r] != 0)]
        if not full_rows:
            return
        n = len(full_rows)
        # Scoring: 100, 300, 500, 800
        points = [0, 100, 300, 500, 800]
        self.score += points[min(n, 4)] * self.level
        self.lines_cleared += n
        self.level = 1 + self.lines_cleared // 10

        # Remove full rows and shift down
        new_grid = np.zeros((ROWS, COLS), dtype=int)
        new_colors = {}
        dest = ROWS - 1
        for src in range(ROWS - 1, -1, -1):
            if src in full_rows:
                continue
            new_grid[dest] = self.grid[src]
            for c in range(COLS):
                if (src, c) in self.grid_colors:
                    new_colors[(dest, c)] = self.grid_colors[(src, c)]
            dest -= 1
        self.grid = new_grid
        self.grid_colors = new_colors

    def get_drop_speed(self):
        """Seconds per gravity tick, decreases with level."""
        return max(0.05, 0.5 - (self.level - 1) * 0.04)


def draw_game(game):
    """Render the game state to an OpenCV image."""
    img = np.full((WIN_H, WIN_W, 3), BG_COLOR, dtype=np.uint8)

    # Draw locked cells
    for r in range(ROWS):
        for c in range(COLS):
            x, y = c * CELL, r * CELL
            if game.grid[r, c] != 0:
                color = PIECE_COLORS.get(game.grid_colors.get((r, c), "I"))
                cv2.rectangle(img, (x + 1, y + 1), (x + CELL - 1, y + CELL - 1), color, -1)

    # Draw current falling piece
    if not game.game_over:
        # Ghost piece (drop preview)
        ghost_row = game.piece_row
        while not game._collides(ghost_row + 1, game.piece_col, game.rotation):
            ghost_row += 1
        if ghost_row != game.piece_row:
            ghost_color = tuple(c // 3 for c in PIECE_COLORS[game.piece])
            for r, c in game._get_cells(ghost_row, game.piece_col, game.rotation):
                if 0 <= r < ROWS and 0 <= c < COLS:
                    x, y = c * CELL, r * CELL
                    cv2.rectangle(img, (x + 1, y + 1), (x + CELL - 1, y + CELL - 1), ghost_color, -1)

        # Actual piece
        color = PIECE_COLORS[game.piece]
        for r, c in game._get_cells(game.piece_row, game.piece_col, game.rotation):
            if 0 <= r < ROWS and 0 <= c < COLS:
                x, y = c * CELL, r * CELL
                cv2.rectangle(img, (x + 1, y + 1), (x + CELL - 1, y + CELL - 1), color, -1)

    # Grid lines
    for r in range(ROWS + 1):
        cv2.line(img, (0, r * CELL), (GAME_W, r * CELL), GRID_COLOR, 1)
    for c in range(COLS + 1):
        cv2.line(img, (c * CELL, 0), (c * CELL, GAME_H), GRID_COLOR, 1)

    # Sidebar
    sx = GAME_W + 10
    cv2.putText(img, "TETRIS", (sx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    cv2.putText(img, f"Score: {game.score}", (sx, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    cv2.putText(img, f"Lines: {game.lines_cleared}", (sx, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    cv2.putText(img, f"Level: {game.level}", (sx, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

    # Next piece preview
    cv2.putText(img, "Next:", (sx, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    next_offsets = SHAPES[game.next_piece][0]
    next_color = PIECE_COLORS[game.next_piece]
    for dr, dc in next_offsets:
        px = sx + 40 + dc * 18
        py = 200 + dr * 18
        cv2.rectangle(img, (px, py), (px + 16, py + 16), next_color, -1)

    # Controls help
    cv2.putText(img, "Controls:", (sx, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    hints = ["Arrows: Move", "Up: Rotate", "Space: Drop", "G: Gestures", "H: Help", "R: Restart", "Q: Quit"]
    for i, h in enumerate(hints):
        cv2.putText(img, h, (sx, 295 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1)

    # Game over overlay
    if game.game_over:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (WIN_W, WIN_H), (0, 0, 0), -1)
        img = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)
        cv2.putText(img, "GAME OVER", (30, GAME_H // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(img, f"Score: {game.score}", (80, GAME_H // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        cv2.putText(img, "R to Restart | Q to Quit", (30, GAME_H // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return img


def draw_help(base_img):
    """Draw a help menu overlay with controls and gesture tips."""
    overlay = base_img.copy()
    cv2.rectangle(overlay, (0, 0), (WIN_W, WIN_H), (0, 0, 0), -1)
    img = cv2.addWeighted(base_img, 0.15, overlay, 0.85, 0)

    title_color = (0, 200, 255)
    head_color = (100, 220, 100)
    text_color = (210, 210, 210)
    tip_color = (180, 180, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cx = WIN_W // 2
    y = 35
    cv2.putText(img, "HELP", (cx - 40, y), font, 0.9, title_color, 2)
    y += 15
    cv2.line(img, (20, y), (WIN_W - 20, y), (80, 80, 80), 1)

    # --- Keyboard Controls ---
    y += 30
    cv2.putText(img, "Keyboard Controls", (20, y), font, 0.5, head_color, 1)
    kb_lines = [
        "Arrow Keys / WASD  -  Move & Rotate",
        "Space              -  Hard Drop",
        "G                  -  Toggle Gestures",
        "H                  -  Toggle Help",
        "R                  -  Restart Game",
        "Q                  -  Quit",
    ]
    for line in kb_lines:
        y += 22
        cv2.putText(img, line, (25, y), font, 0.35, text_color, 1)

    # --- Gesture Controls ---
    y += 35
    cv2.putText(img, "Gesture Controls", (20, y), font, 0.5, head_color, 1)
    gesture_lines = [
        "Pinch thumb + index    ->  Move Left",
        "Pinch thumb + middle   ->  Move Right",
        "Thumb up (only)        ->  Rotate",
        "Closed fist            ->  Hard Drop",
    ]
    for line in gesture_lines:
        y += 22
        cv2.putText(img, line, (25, y), font, 0.35, text_color, 1)

    # --- Tips ---
    y += 35
    cv2.putText(img, "Gesture Tips", (20, y), font, 0.5, head_color, 1)
    tips = [
        "* Open your hand slightly between",
        "  pinches to reset detection.",
        "* Keep your hand steady for the",
        "  camera to track it clearly.",
        "* Use quick, deliberate pinches",
        "  rather than slow squeezes.",
        "* Fist has a long cooldown -- use",
        "  it only when you're sure.",
        "* Hold your hand 30-60 cm from",
        "  the webcam for best tracking.",
    ]
    for line in tips:
        y += 20
        cv2.putText(img, line, (25, y), font, 0.33, tip_color, 1)

    y += 30
    cv2.putText(img, "Press H to close", (cx - 60, y), font, 0.4, (120, 120, 120), 1)

    return img


class GestureController:
    """MediaPipe hand gesture detection for Tetris controls (tasks API)."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

    # Hand landmark indices
    WRIST = 0
    THUMB_TIP, THUMB_IP = 4, 3
    INDEX_TIP, INDEX_PIP = 8, 6
    MIDDLE_TIP, MIDDLE_PIP = 12, 10
    RING_TIP, RING_PIP = 16, 14
    PINKY_TIP, PINKY_PIP = 20, 18

    # Landmark connections for drawing (matching mediapipe hand connections)
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),        # thumb
        (0,5),(5,6),(6,7),(7,8),        # index
        (5,9),(9,10),(10,11),(11,12),   # middle
        (9,13),(13,14),(14,15),(15,16), # ring
        (13,17),(17,18),(18,19),(19,20),# pinky
        (0,17),                         # palm
    ]

    def __init__(self):
        self.cap = None
        self.landmarker = None
        self.last_action_time = {}
        self.cooldowns = {
            "move_left": 0.25,
            "move_right": 0.25,
            "rotate": 0.6,
            "hard_drop": 1.0,
        }
        self.pinch_threshold = 0.07      # distance to count as "touching"
        self.pinch_fired = False         # prevents repeat until hand opens

    def _ensure_model(self):
        if os.path.exists(self.MODEL_FILE):
            return
        print(f"Downloading hand landmarker model to {self.MODEL_FILE}...")
        import urllib.request
        urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_FILE)
        print("Download complete.")

    def start(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        if self.landmarker is None:
            self._ensure_model()
            import mediapipe as mp
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.MODEL_FILE),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            self.landmarker = HandLandmarker.create_from_options(options)
            self._frame_ts = 0

    def stop(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        cv2.destroyWindow("Webcam - Gesture Control")

    def _cooled_down(self, action):
        now = time.time()
        last = self.last_action_time.get(action, 0)
        if now - last >= self.cooldowns[action]:
            self.last_action_time[action] = now
            return True
        return False

    def _fingers_up(self, landmarks):
        """Return [thumb, index, middle, ring, pinky] booleans."""
        tips = [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        pips = [self.THUMB_IP, self.INDEX_PIP, self.MIDDLE_PIP, self.RING_PIP, self.PINKY_PIP]
        fingers = []
        # Thumb: tip x vs ip x (for right hand facing camera = mirrored)
        fingers.append(landmarks[tips[0]].x < landmarks[pips[0]].x)
        # Others: tip y < pip y means extended
        for i in range(1, 5):
            fingers.append(landmarks[tips[i]].y < landmarks[pips[i]].y)
        return fingers

    def _draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks and connections on the frame."""
        h, w = frame.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for i, j in self.CONNECTIONS:
            cv2.line(frame, points[i], points[j], (0, 200, 0), 2)
        for pt in points:
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    def process(self, game):
        """Read one webcam frame, detect gesture, apply action. Returns frame or None."""
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)

        # Convert to mediapipe Image and detect
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._frame_ts += 33  # ~30fps in milliseconds
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        action_text = "No hand"

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            self._draw_landmarks(frame, landmarks)

            wrist_x = landmarks[self.WRIST].x
            fingers = self._fingers_up(landmarks)

            # Closed fist: no fingers up -> hard drop
            if not any(fingers):
                action_text = "FIST -> Hard Drop"
                if self._cooled_down("hard_drop"):
                    game.hard_drop()

            # Thumb only up -> rotate
            elif fingers[0] and not any(fingers[1:]):
                action_text = "THUMB -> Rotate"
                if self._cooled_down("rotate"):
                    game.rotate()

            # Pinch detection: thumb+index = left, thumb+middle = right
            else:
                thumb = landmarks[self.THUMB_TIP]
                index = landmarks[self.INDEX_TIP]
                middle = landmarks[self.MIDDLE_TIP]
                d_index = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2) ** 0.5
                d_middle = ((thumb.x - middle.x)**2 + (thumb.y - middle.y)**2) ** 0.5

                if not self.pinch_fired:
                    if d_index < self.pinch_threshold:
                        action_text = "PINCH INDEX -> Left"
                        if self._cooled_down("move_left"):
                            game.move(0, -1)
                        self.pinch_fired = True
                    elif d_middle < self.pinch_threshold:
                        action_text = "PINCH MIDDLE -> Right"
                        if self._cooled_down("move_right"):
                            game.move(0, 1)
                        self.pinch_fired = True
                    else:
                        action_text = "Open hand (ready)"
                else:
                    # Reset when both fingers are away from thumb
                    if d_index > self.pinch_threshold * 1.2 and d_middle > self.pinch_threshold * 1.2:
                        self.pinch_fired = False
                        action_text = "Open hand (ready)"
                    else:
                        action_text = "Open hand (reset to move)"
        else:
            self.pinch_fired = False

        # Draw action label on frame
        cv2.putText(frame, action_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Pinch Idx=L Mid=R | Fist=Drop | Thumb=Rot", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    def release(self):
        self.stop()
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None


def main():
    game = TetrisGame()
    gesture = GestureController()
    gesture_mode = False
    show_help = False

    last_drop = time.time()
    window_name = "Tetris"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        now = time.time()

        # Gesture input
        if gesture_mode:
            cam_frame = gesture.process(game)
            if cam_frame is not None:
                cv2.imshow("Webcam - Gesture Control", cam_frame)

        # Gravity
        if not game.game_over and now - last_drop >= game.get_drop_speed():
            game.step()
            last_drop = now

        # Render
        frame = draw_game(game)
        if gesture_mode:
            cv2.putText(frame, "[GESTURE ON]", (GAME_W + 10, WIN_H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        if show_help:
            frame = draw_help(frame)
        cv2.imshow(window_name, frame)

        # Keyboard input
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            game.reset()
            last_drop = time.time()
        elif key == ord("h"):
            show_help = not show_help
        elif key == ord("g"):
            gesture_mode = not gesture_mode
            if gesture_mode:
                gesture.start()
            else:
                gesture.stop()
        elif not game.game_over:
            # Arrow keys: OpenCV returns 0-3 on some platforms, or special codes
            # Also support WASD as fallback
            if key in (81, 2, ord("a")):       # left arrow / A
                game.move(0, -1)
            elif key in (83, 3, ord("d")):     # right arrow / D
                game.move(0, 1)
            elif key in (84, 1, ord("s")):     # down arrow / S
                game.move(1, 0)
            elif key in (82, 0, ord("w")):     # up arrow / W
                game.rotate()
            elif key == ord(" "):
                game.hard_drop()

    gesture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
