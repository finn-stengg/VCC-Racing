"""
Autonomous Vehicle Client

A client for connecting to and controlling an autonomous vehicle simulation,
retrieving state data and camera images (raw and segmented).
"""

import socket
import json
import argparse
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import cv2
import numpy as np


# =========================
# GLOBAL STATE (RATE LIMIT)
# =========================
prev_steer = 0.0


@dataclass
class CommandMode:
    """Command mode constants for vehicle communication."""
    STATE_DATA = 185  # Request vehicle state data
    RAW_IMAGE = 203   # Request raw camera image
    SEGMENTED_IMAGE = 31  # Request segmented camera image


class AVClientError(Exception):
    """Base exception for AV Client errors."""
    pass


class ConnectionError(AVClientError):
    """Raised when connection to server fails."""
    pass


class DataRetrievalError(AVClientError):
    """Raised when data retrieval fails."""
    pass


class AVClient:
    """
    Client for autonomous vehicle simulation control and data retrieval.

    Manages connection to vehicle server, sends control commands,
    and retrieves state data and camera images.
    """

    MAX_DGRAM = 2**16  # Maximum datagram size for socket operations

    def __init__(self, host: str = '127.0.0.1', port: int = 11000,
                 timeout: float = 5.0):
        """
        Initialize AV Client.

        Args:
            host: Server hostname or IP address
            port: Server port number
            timeout: Socket timeout in seconds (default: 5.0)
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None
        self.speed_cmd = 0
        self.angle_cmd = 0

    def connect(self) -> None:
        """
        Establish connection to vehicle server.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            print(f"Connected to vehicle server at {self.host}:{self.port}")
        except socket.error as e:
            raise ConnectionError(
                f"Failed to connect to {self.host}:{self.port}: {e}"
            )

    def disconnect(self) -> None:
        """Close connection to vehicle server."""
        if self.socket:
            try:
                self.socket.close()
                print("Disconnected from vehicle server")
            except socket.error as e:
                print(f"Error closing socket: {e}")
            finally:
                self.socket = None

    def __enter__(self):
        """Context manager entry - establish connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up connection."""
        self.disconnect()
        return False

    def set_control(self, speed: int, angle: int) -> None:
        """
        Set vehicle control commands.

        Args:
            speed: Speed command (max 90)
            angle: Steering angle command (max ±25)
        """
        self.speed_cmd = speed
        self.angle_cmd = angle

    def _create_command_json(self, mode: int) -> bytes:
        """
        Create JSON command for server communication.

        Args:
            mode: Command mode (STATE_DATA, RAW_IMAGE, or SEGMENTED_IMAGE)

        Returns:
            JSON command as bytes
        """
        cmd: Dict[str, Any] = {'Cmd': mode}

        if mode == CommandMode.STATE_DATA:
            cmd['Speed'] = self.speed_cmd
            cmd['Angle'] = self.angle_cmd

        return json.dumps(cmd).encode('utf-8')

    def _receive_data(self, expected_length: int) -> bytes:
        """
        Receive data from socket with length validation.

        Args:
            expected_length: Expected number of bytes to receive

        Returns:
            Received data bytes

        Raises:
            DataRetrievalError: If data reception fails
        """
        if not self.socket:
            raise DataRetrievalError("Not connected to server")

        data = b''
        try:
            while len(data) < expected_length:
                to_read = expected_length - len(data)
                chunk = self.socket.recv(min(self.MAX_DGRAM, to_read))
                if not chunk:
                    raise DataRetrievalError("Connection closed by server")
                data += chunk
        except socket.error as e:
            raise DataRetrievalError(
                f"Socket error during data reception: {e}"
            )

        return data

    def get_state_data(self) -> Dict[str, Any]:
        """
        Retrieve current vehicle state data.

        Returns:
            Dictionary containing vehicle state information

        Raises:
            DataRetrievalError: If state data retrieval fails
        """
        if not self.socket:
            raise DataRetrievalError("Not connected to server")

        try:
            # Send state data request
            self.socket.sendall(
                self._create_command_json(CommandMode.STATE_DATA)
            )

            # Receive data length
            length_bytes = self.socket.recv(8)
            if len(length_bytes) != 8:
                raise DataRetrievalError("Failed to receive data length")

            data_length = int.from_bytes(length_bytes, "big")

            # Receive state data
            data = self._receive_data(data_length)
            return json.loads(data)

        except socket.timeout:
            raise DataRetrievalError(
                "Timeout waiting for state data from server"
            )
        except json.JSONDecodeError as e:
            raise DataRetrievalError(f"Invalid JSON response: {e}")
        except socket.error as e:
            raise DataRetrievalError(f"Socket error: {e}")

    def get_image(self, mode: int) -> np.ndarray:
        """
        Retrieve camera image from vehicle.

        Args:
            mode: Image mode (RAW_IMAGE or SEGMENTED_IMAGE)

        Returns:
            Image as numpy array

        Raises:
            DataRetrievalError: If image retrieval fails
            ValueError: If invalid mode specified
        """
        if mode not in (CommandMode.RAW_IMAGE, CommandMode.SEGMENTED_IMAGE):
            raise ValueError(f"Invalid image mode: {mode}")

        if not self.socket:
            raise DataRetrievalError("Not connected to server")

        image_type = "raw" if mode == CommandMode.RAW_IMAGE else "segmented"

        try:
            # Send image request
            self.socket.sendall(self._create_command_json(mode))

            # Receive data length
            length_bytes = self.socket.recv(8)
            if len(length_bytes) != 8:
                raise DataRetrievalError("Failed to receive data length")

            data_length = int.from_bytes(length_bytes, "big")

            # Receive image data
            data = self._receive_data(data_length)

            # Decode image
            image = cv2.imdecode(
                np.frombuffer(data, np.uint8),
                cv2.IMREAD_UNCHANGED
            )

            if image is None:
                raise DataRetrievalError("Failed to decode image data")

            return image

        except socket.timeout:
            raise DataRetrievalError(
                f"Timeout waiting for {image_type} image from server"
            )
        except socket.error as e:
            raise DataRetrievalError(f"Socket error: {e}")

    def get_raw_image(self) -> np.ndarray:
        """
        Retrieve raw camera image.

        Returns:
            Raw camera image as numpy array
        """
        return self.get_image(CommandMode.RAW_IMAGE)

    def get_segmented_image(self) -> np.ndarray:
        """
        Retrieve segmented camera image.

        Returns:
            Segmented camera image as numpy array
        """
        return self.get_image(CommandMode.SEGMENTED_IMAGE)


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Autonomous Vehicle Client - Connect to AV simulation server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py                    # Connect to default 127.0.0.1:11000
  python client.py 11001              # Connect to 127.0.0.1:11001
  python client.py --host 192.168.1.10 --port 11000
  python client.py --timeout 10       # Set 10 second timeout
        """
    )

    parser.add_argument(
        'port',
        nargs='?',
        type=int,
        default=11000,
        help='Server port (default: 11000)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Server hostname or IP address (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--timeout',
        type=float,
        default=5.0,
        help='Socket timeout in seconds (default: 5.0)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Disable debug messages'
    )

    return parser.parse_args()


def calculate_steering_angle(
    segmented_image: np.ndarray,
    speed_cmd: int
) -> Tuple[int, np.ndarray]:

    MAX_ANGLE = 25

    LOWER = np.array([240, 10, 165], dtype=np.uint8)
    UPPER = np.array([255, 35, 195], dtype=np.uint8)

    HEADING_DEADZONE_DEG = 2.0
    EMA_ALPHA = 0.18
    M00_THRESH = 600

    STRAIGHT_HEADING_DEG = 2.0
    STRAIGHT_DX_PX = 6
    STRAIGHT_FRAMES = 6

    # ======================
    # INIT STATE
    # ======================
    if not hasattr(calculate_steering_angle, "_prev_steer"):
        calculate_steering_angle._prev_steer = 0.0
        calculate_steering_angle._last_band = "mid"
        calculate_steering_angle._corner_frames = 0
        calculate_steering_angle._straight_count = 0

    prev_steer = calculate_steering_angle._prev_steer
    last_band = calculate_steering_angle._last_band

    h, w = segmented_image.shape[:2]
    cx_img = w // 2
    vis = segmented_image.copy()

    # ======================
    # MASK
    # ======================
    mask = cv2.inRange(segmented_image, LOWER, UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    cv2.line(vis, (cx_img, 0), (cx_img, h), (0, 255, 0), 2)

    # ======================
    # MULTI-BAND
    # ======================
    bands = {
        "near": (int(h * 0.65), int(h * 0.80)),
        "mid":  (int(h * 0.45), int(h * 0.65)),
        "far":  (int(h * 0.25), int(h * 0.45)),
    }

    candidates = {}
    for name, (y1, y2) in bands.items():
        band = mask[y1:y2, :]
        m = cv2.moments(band, binaryImage=True)
        if m["m00"] > M00_THRESH:
            cx = int(m["m10"] / m["m00"])
            cy = (y1 + y2) // 2
            candidates[name] = (cx, cy, m["m00"])

    if not candidates:
        return int(round(prev_steer)), vis

    # ======================
    # BAND SELECTION
    # ======================
    if last_band in candidates:
        target_band = last_band
    else:
        target_band = max(candidates, key=lambda k: candidates[k][2])

    if speed_cmd >= 28 and "mid" in candidates:
        target_band = "mid"

    calculate_steering_angle._last_band = target_band
    cx, cy, m00 = candidates[target_band]

    # ======================
    # GEOMETRY (MID)
    # ======================
    dx = cx - cx_img
    dy = max(h - cy, 1)
    heading_deg = np.degrees(np.arctan2(dx, dy))

    # ======================
    # FAR LOOKAHEAD (FORK AWARENESS)
    # ======================
    far_heading = 0.0
    if "far" in candidates:
        fx, fy, _ = candidates["far"]
        far_heading = np.degrees(np.arctan2(fx - cx_img, max(h - fy, 1)))

    is_fork = abs(far_heading) > 5.0

    # ======================
    # CONTEXT STATE
    # ======================
    is_corner = abs(heading_deg) > 6 or abs(prev_steer) > 8

    if is_corner:
        calculate_steering_angle._corner_frames = 12
    else:
        calculate_steering_angle._corner_frames = max(
            calculate_steering_angle._corner_frames - 1, 0
        )

    if abs(heading_deg) < STRAIGHT_HEADING_DEG and abs(dx) < STRAIGHT_DX_PX:
        calculate_steering_angle._straight_count += 1
    else:
        calculate_steering_angle._straight_count = 0

    # ======================
    # DEADZONE
    # ======================
    effective_deadzone = (
        3.5 if calculate_steering_angle._corner_frames > 0
        else HEADING_DEADZONE_DEG
    )
    if abs(heading_deg) < effective_deadzone:
        heading_deg = 0.0

    # ======================
    # GAIN
    # ======================
    if speed_cmd <= 15:
        k = 0.6
    elif speed_cmd <= 25:
        k = 0.45
    else:
        k = 0.38

    steer_raw = np.clip(k * heading_deg, -MAX_ANGLE, MAX_ANGLE)

    # ======================
    # EMA
    # ======================
    steer_ema = EMA_ALPHA * steer_raw + (1 - EMA_ALPHA) * prev_steer

    # ======================
    # RATE LIMIT
    # ======================
    max_delta = 1.0 if speed_cmd >= 26 else 2.0
    steer_final = np.clip(
        steer_ema,
        prev_steer - max_delta,
        prev_steer + max_delta
    )

    # ======================
    # POST-CORNER DAMPING
    # ======================
    if calculate_steering_angle._corner_frames > 0:
        steer_final *= 0.85

    steer_final = np.clip(steer_final, -MAX_ANGLE, MAX_ANGLE)

    # ======================
    # STRAIGHT LOCK (MAP 2 SAFE)
    # ======================
    if (calculate_steering_angle._straight_count >= STRAIGHT_FRAMES
            and not is_fork):
        steer_final = np.clip(steer_final, -1.5, 1.5)

    # ======================
    # UPDATE STATE
    # ======================
    calculate_steering_angle._prev_steer = steer_final
    steer_int = int(round(steer_final))

    # ======================
    # DEBUG
    # ======================
    cv2.circle(vis, (cx, cy), 7, (0, 0, 255), -1)
    cv2.arrowedLine(vis, (cx_img, h - 5), (cx, cy), (0, 255, 255), 2)

    cv2.putText(
        vis, f"Angle: {steer_int:+d} (raw {steer_raw:+.1f})",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    cv2.putText(
        vis, f"Heading: {heading_deg:+.1f} deg",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        vis, f"Target: {target_band}  m00={int(m00)}",
        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    return steer_int, vis


def detect_fork(segmented_image):
    h, w = segmented_image.shape[:2]
    cx = w // 2

    # dùng MID band (ổn định hơn)
    y1 = int(h * 0.40)
    y2 = int(h * 0.60)

    mask = cv2.inRange(
        segmented_image,
        np.array([240, 10, 165], np.uint8),
        np.array([255, 35, 195], np.uint8)
    )

    band = mask[y1:y2, :]
    m = cv2.moments(band, binaryImage=True)

    if m["m00"] < 1200:   # nới ngưỡng
        return False, 0.0

    fx = int(m["m10"] / m["m00"])
    dx = fx - cx
    heading = np.degrees(np.arctan2(dx, h - (y1 + y2) // 2))

    if abs(dx) > 18:
        return True, heading
    return False, heading


def detect_sign(segmented_image: np.ndarray) -> str:
    """
    Detect traffic sign using semantic segmented image.
    Detect LEFT / RIGHT / STRAIGHT
    """
    h, w = segmented_image.shape[:2]

    # ROI phía trước – bên phải
    x1 = int(w * 0.60)
    x2 = int(w * 0.95)
    y1 = int(h * 0.25)
    y2 = int(h * 0.55)

    roi = segmented_image[y1:y2, x1:x2]

    # ===== COLOR DEFINITIONS (BGR) =====

    # LEFT sign (cyan)
    LEFT_LO = np.array([200, 200,   0], np.uint8)
    LEFT_HI = np.array([255, 255,  80], np.uint8)

    # STRAIGHT sign (purple)
    STRAIGHT_LO = np.array([200,   0, 200], np.uint8)
    STRAIGHT_HI = np.array([255,  80, 255], np.uint8)

    # RIGHT sign (light pink / light purple)
    RIGHT_LO = np.array([160, 160, 230], np.uint8)
    RIGHT_HI = np.array([210, 210, 255], np.uint8)

    # ===== MASK =====
    mask_left = cv2.inRange(roi, LEFT_LO, LEFT_HI)
    mask_straight = cv2.inRange(roi, STRAIGHT_LO, STRAIGHT_HI)
    mask_right = cv2.inRange(roi, RIGHT_LO, RIGHT_HI)

    left_count = cv2.countNonZero(mask_left)
    straight_count = cv2.countNonZero(mask_straight)
    right_count = cv2.countNonZero(mask_right)

    PIXEL_THRESH = 120

    if left_count > PIXEL_THRESH:
        return "LEFT"

    if right_count > PIXEL_THRESH:
        return "RIGHT"

    if straight_count > PIXEL_THRESH:
        return "STRAIGHT"

    return "NONE"


def lane_available(mask: np.ndarray, direction: str, h: int, w: int):
    """
    Return (is_open: bool, area: int)
    """
    y1 = int(h * 0.45)
    y2 = int(h * 0.75)

    if direction == "LEFT":
        x1, x2 = 0, int(w * 0.40)
    elif direction == "RIGHT":
        x1, x2 = int(w * 0.60), w
    elif direction == "STRAIGHT":
        x1, x2 = int(w * 0.30), int(w * 0.70)
    else:
        return False, 0

    roi = mask[y1:y2, x1:x2]
    area = cv2.countNonZero(roi)
    return (area > 500), area


def apply_turn_control(turn_intent, speed_cmd, steer_lane,
                       fork_timer=None, auto=False):
    MAX_ANGLE = 25
    STEER_POLARITY = -1

    if auto:
        # AUTO-FORK (Map 1): vào cua sớm & mạnh
        if turn_intent == "LEFT":
            steer_cmd = STEER_POLARITY * (+14)
        elif turn_intent == "RIGHT":
            steer_cmd = STEER_POLARITY * (-14)
        else:
            steer_cmd = steer_lane

        speed_cmd = min(speed_cmd, 16)
        return speed_cmd, int(np.clip(steer_cmd, -MAX_ANGLE, MAX_ANGLE))

    # ===== SIGN-BASED (Map 2): ramp mượt =====
    def ramp(timer_left, t_max):
        if timer_left is None:
            return 12
        p = 1.0 - (timer_left / max(t_max, 1))
        if p < 0.35:
            return 8
        elif p < 0.70:
            return 10
        else:
            return 12

    if turn_intent == "LEFT":
        mag = ramp(fork_timer, 22)
        steer_cmd = STEER_POLARITY * (+mag)
        speed_cmd = min(speed_cmd, 18)

    elif turn_intent == "RIGHT":
        mag = ramp(fork_timer, 22)
        steer_cmd = STEER_POLARITY * (-mag)
        speed_cmd = min(speed_cmd, 18)

    else:
        steer_cmd = steer_lane

    return speed_cmd, int(np.clip(steer_cmd, -MAX_ANGLE, MAX_ANGLE))


def is_near_intersection(segmented_image):
    h, w = segmented_image.shape[:2]
    roi = segmented_image[int(h * 0.55):int(h * 0.75),
                          int(w * 0.35):int(w * 0.65)]

    purple_mask = cv2.inRange(
        roi,
        np.array([240, 10, 165], np.uint8),
        np.array([255, 35, 195], np.uint8)
    )
    area = cv2.countNonZero(purple_mask)
    return (area > 1200), area


def detect_fork_early(segmented_image):
    h, w = segmented_image.shape[:2]
    cx = w // 2

    y1 = int(h * 0.25)
    y2 = int(h * 0.40)   # FAR band

    mask = cv2.inRange(
        segmented_image,
        np.array([240, 10, 165], np.uint8),
        np.array([255, 35, 195], np.uint8)
    )

    band = mask[y1:y2, :]
    m = cv2.moments(band, binaryImage=True)

    if m["m00"] < 600:
        return False, 0.0

    fx = int(m["m10"] / m["m00"])
    dx = fx - cx
    heading = np.degrees(np.arctan2(dx, h - (y1 + y2) // 2))

    return abs(dx) > 10, heading


def main():
    args = parse_arguments()
    print(f"[CONFIG] Connecting to {args.host}:{args.port} "
          f"(timeout: {args.timeout}s)")

    # =========================
    # FORK STATE MACHINE
    # =========================
    fork_state = "IDLE"          # IDLE | PREPARE | COMMIT
    fork_intent = "NONE"         # NONE | LEFT | RIGHT
    fork_timer = 0

    # sign hold
    sign_hold = 0
    SIGN_HOLD_FRAMES = 40

    # PREPARE timing
    prepare_cnt = 0
    PREPARE_MIN = 8
    PREPARE_MAX = 80

    # intersection hysteresis (still used for sign-based turns)
    inter_hold = 0
    inter_lost = 0
    INTER_HOLD_FRAMES = 28
    INTER_LOST_CANCEL = 6

    # block logic
    LEFT_BLOCK_STEER = +2
    RIGHT_BLOCK_STEER = -6

    # opening threshold
    LEFT_STRONG_AREA = 1100
    RIGHT_STRONG_AREA = 400

    # geometry threshold
    FORK_LEFT_HEADING = -4.0
    FORK_RIGHT_HEADING = +0.5

    INTER_AREA_STRONG = 1800
    inter_seen = 0

    # =========================
    # STEP 1: AUTO-FORK (NO SIGN) - T junction detector
    # =========================
    steer_hist = deque(maxlen=10)

    # Ngã 3 detector bằng "straight đóng + 1/2 bên mở"
    T_STRAIGHT_MAX = 380          # straight_area nhỏ => phía trước không còn lane
    T_SIDE_MIN = 850              # trái/phải đủ mở
    t_seen = 0

    # Post-commit recovery (giảm văng sau DONE)
    recover_timer = 0
    RECOVER_FRAMES = 10
    RECOVER_STEER_CLAMP = 3
    RECOVER_SPEED_MAX = 24

    fork_cooldown = 0
    FORK_COOLDOWN_FRAMES = 45   # ~0.75s @60FPS

    try:
        with AVClient(host=args.host, port=args.port,
                      timeout=args.timeout) as client:
            print("[INFO] Starting main loop. Press 'q' to exit.\n")

            while True:
                # ======================
                # 1. GET DATA
                # ======================
                _state = client.get_state_data()
                raw_image = client.get_raw_image()
                segmented_image = client.get_segmented_image()
                h, w = segmented_image.shape[:2]

                # ======================
                # 2. COMMON LANE MASK
                # ======================
                LOWER = np.array([240, 10, 165], np.uint8)
                UPPER = np.array([255, 35, 195], np.uint8)

                mask = cv2.inRange(segmented_image, LOWER, UPPER)
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
                )
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
                )

                # ======================
                # 3. SPEED PLANNER
                # ======================
                speed_cmd = 30
                steer_preview, _ = calculate_steering_angle(
                    segmented_image, speed_cmd
                )
                a = abs(steer_preview)

                if a < 5:
                    speed_cmd = 30
                elif a < 10:
                    speed_cmd = 22
                elif a < 15:
                    speed_cmd = 16
                else:
                    speed_cmd = 12

                # ======================
                # 4. LANE FOLLOW
                # ======================
                steer_lane, vis = calculate_steering_angle(
                    segmented_image, speed_cmd
                )
                steer_hist.append(float(steer_lane))

                # ======================
                # 5. SIGN → INTENT HOLD
                # ======================
                sign = detect_sign(segmented_image)

                if sign in ("LEFT", "RIGHT"):
                    fork_intent = sign
                    sign_hold = SIGN_HOLD_FRAMES
                else:
                    sign_hold = max(sign_hold - 1, 0)
                    if sign_hold == 0 and fork_state == "IDLE":
                        fork_intent = "NONE"

                # ======================
                # 6. INTERSECTION GATE (still for sign-based)
                # ======================
                near_inter, inter_area = is_near_intersection(segmented_image)

                if near_inter:
                    inter_seen = min(inter_seen + 1, 50)
                else:
                    inter_seen = max(inter_seen - 1, 0)

                if near_inter:
                    inter_hold = INTER_HOLD_FRAMES
                else:
                    inter_hold = max(inter_hold - 1, 0)

                near_gate = (near_inter or inter_hold > 0)

                # ======================
                # 7. OPENING + GEOMETRY
                # ======================
                left_open, left_area = lane_available(mask, "LEFT", h, w)
                right_open, right_area = lane_available(mask, "RIGHT", h, w)
                straight_open, straight_area = lane_available(
                    mask, "STRAIGHT", h, w
                )

                fork_mid_ok, fork_mid_heading = detect_fork(segmented_image)
                fork_far_ok, fork_far_heading = detect_fork_early(
                    segmented_image
                )

                # ======================
                # STEP 1: T-JUNCTION DETECTOR (NO SIGN)
                # ======================
                # T-junction: phía trước (straight) gần như mất,
                # nhưng bên trái/phải mở
                is_t_junction = (
                    (straight_area <= T_STRAIGHT_MAX)
                    and ((left_area >= T_SIDE_MIN)
                         or (right_area >= T_SIDE_MIN))
                )

                if is_t_junction:
                    t_seen = min(t_seen + 1, 20)
                else:
                    t_seen = max(t_seen - 1, 0)

                # ======================
                # 8. FORK STATE MACHINE
                # ======================
                if fork_state == "IDLE":
                    prepare_cnt = 0
                    inter_lost = 0

                    # ===== AUTO-FORK (NO SIGN) =====
                    if fork_intent == "NONE":
                        # Chỉ khi đang ở gần intersection
                        # (đã thấy ít nhất 2 frame)
                        if inter_seen >= 1 and near_gate:
                            # Ngã 3 chữ T: đường thẳng gần như đóng
                            T_STRAIGHT_MAX = 250
                            SIDE_STRONG = 700
                            SIDE_DOMINATE = 2.5

                            straight_open, straight_area = lane_available(
                                mask, "STRAIGHT", h, w
                            )

                            is_T = (straight_area <= T_STRAIGHT_MAX)

                            # Ưu tiên nhánh nào "thắng" rõ
                            left_strong = (left_area >= SIDE_STRONG)
                            right_strong = (right_area >= SIDE_STRONG)

                            left_dom = left_area >= right_area * SIDE_DOMINATE
                            right_dom = right_area >= left_area * SIDE_DOMINATE

                            if is_T and (left_strong or right_strong):
                                if left_dom and left_open:
                                    fork_state = "COMMIT"
                                    fork_intent = "LEFT"
                                    fork_timer = 14
                                    print(
                                        f"[FORK] AUTO COMMIT LEFT (T-junction, "
                                        f"L={left_area}, R={right_area}, "
                                        f"S={straight_area})"
                                    )

                                elif right_dom and right_open:
                                    fork_state = "COMMIT"
                                    fork_intent = "RIGHT"
                                    fork_timer = 14
                                    print(
                                        f"[FORK] AUTO COMMIT RIGHT (T-junction, "
                                        f"L={left_area}, R={right_area}, "
                                        f"S={straight_area})"
                                    )

                    # ===== SIGN-BASED PREPARE =====
                    if fork_state == "IDLE":
                        if fork_intent in ("LEFT", "RIGHT") and near_gate:
                            fork_state = "PREPARE"
                            print(f"[FORK] PREPARE intent={fork_intent}")

                elif fork_state == "PREPARE":
                    prepare_cnt += 1

                    if not near_gate:
                        inter_lost += 1
                    else:
                        inter_lost = 0

                    if inter_lost >= INTER_LOST_CANCEL:
                        fork_state = "IDLE"
                        prepare_cnt = 0
                        inter_lost = 0
                        print("[FORK] CANCEL (lost intersection)")
                        continue

                    if prepare_cnt > PREPARE_MAX:
                        fork_state = "IDLE"
                        prepare_cnt = 0
                        print("[FORK] CANCEL (timeout)")
                        continue

                    prepare_min = PREPARE_MIN
                    if fork_intent == "RIGHT":
                        prepare_min = 14
                    ready = (prepare_cnt >= prepare_min)

                    # -------- LEFT --------
                    if fork_intent == "LEFT":
                        if steer_lane <= LEFT_BLOCK_STEER:
                            ok_geom = (
                                fork_mid_ok
                                and (fork_mid_heading <= FORK_LEFT_HEADING)
                            )
                            ok_open = (
                                left_open
                                and (left_area >= LEFT_STRONG_AREA)
                            )

                            if ready and ok_geom and ok_open:
                                fork_state = "COMMIT"
                                fork_timer = 24
                                print(
                                    f"[FORK] COMMIT LEFT (areaL={left_area}, "
                                    f"heading={fork_mid_heading:+.1f})"
                                )

                    # -------- RIGHT --------
                    elif fork_intent == "RIGHT":
                        if steer_lane >= RIGHT_BLOCK_STEER:
                            ok_geom = (
                                fork_far_ok
                                and (fork_far_heading >= FORK_RIGHT_HEADING)
                            )
                            ok_open = (
                                right_open
                                and (right_area >= RIGHT_STRONG_AREA)
                            )
                            very_open = (
                                right_open
                                and (right_area >= RIGHT_STRONG_AREA * 2)
                                and (inter_area >= INTER_AREA_STRONG)
                            )

                            bias_ok = (steer_lane >= 2)
                            if ready and ((ok_geom and ok_open)
                                          or (very_open and bias_ok)):
                                fork_state = "COMMIT"
                                fork_timer = 20
                                print(
                                    f"[FORK] COMMIT RIGHT "
                                    f"(areaR={right_area}, "
                                    f"heading={fork_far_heading:+.1f}, "
                                    f"geom={ok_geom}, very_open={very_open})"
                                )

                elif fork_state == "COMMIT":
                    fork_timer -= 1
                    if fork_timer <= 0:
                        fork_state = "IDLE"
                        fork_intent = "NONE"
                        sign_hold = 0
                        prepare_cnt = 0
                        inter_lost = 0
                        fork_cooldown = FORK_COOLDOWN_FRAMES
                        print("[FORK] DONE")

                # ======================
                # 9. PRE-BIAS RIGHT (NHẸ)
                # ======================
                if fork_state == "PREPARE" and fork_intent == "RIGHT":
                    steer_lane = max(steer_lane, +1)

                # ======================
                # 10. CONTROL MERGE
                # ======================
                if fork_state == "COMMIT":
                    speed_cmd, steer_cmd = apply_turn_control(
                        fork_intent,
                        speed_cmd,
                        steer_lane,
                        fork_timer,
                        auto=(sign == "NONE")
                    )
                else:
                    steer_cmd = steer_lane

                # post-commit recovery clamp
                if recover_timer > 0 and fork_state == "IDLE":
                    recover_timer -= 1
                    speed_cmd = min(speed_cmd, RECOVER_SPEED_MAX)
                    steer_cmd = int(
                        np.clip(steer_cmd,
                                -RECOVER_STEER_CLAMP,
                                RECOVER_STEER_CLAMP)
                    )

                # ======================
                # 11. SEND CONTROL
                # ======================
                print(
                    f"[CONTROL] speed={speed_cmd}, steer={steer_cmd}, "
                    f"sign={sign}, fork={fork_state}"
                )
                client.set_control(speed=speed_cmd, angle=steer_cmd)

                # ======================
                # 12. DEBUG
                # ======================
                cv2.imshow("Raw Camera", raw_image)
                cv2.imshow("Lane + Steering", vis)

                if cv2.waitKey(1) == ord('q'):
                    print("[INFO] Exit requested")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except AVClientError as e:
        print(f"AV Client error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()