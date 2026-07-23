# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UvcZedCamera — read a ZED stereo camera as a plain UVC device via OpenCV.

The ZED (2i, etc.) exposes itself as a standard UVC webcam that delivers a
single *side-by-side* stereo frame (e.g. 2560x720 for HD720-per-eye). This
driver opens it with ``cv2.VideoCapture``, crops to one eye, and returns the
per-eye color image — no ZED SDK required. It implements the LeRobot
``Camera`` protocol so it plugs into the generic ``cameras`` dict.
"""

import threading
import time
from threading import Thread

import numpy as np

from lerobot.common.robot_devices.cameras.configs import UvcZedCameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)
from lerobot.common.utils.utils import capture_timestamp_utc


def _device_index(device: str | int) -> int:
    """Map '/dev/videoN' or 'N' (or an int) to the integer OpenCV/V4L2 index."""
    if isinstance(device, int):
        return device
    device = str(device)
    if device.startswith("/dev/video"):
        return int(device.replace("/dev/video", ""))
    return int(device)


class UvcZedCamera:
    """OpenCV/V4L2 client for a ZED camera in UVC mode (single eye output)."""

    def __init__(self, config: UvcZedCameraConfig):
        self.config = config
        self.device = config.device
        self.device_index = _device_index(config.device)
        # Per-eye (output) resolution that will be recorded.
        self.width = config.width
        self.height = config.height
        self.fps = config.fps
        self.color_mode = config.color_mode
        self.eye = config.eye
        self.rotation = config.rotation
        self.mock = config.mock

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}

        # Map rotation degrees to cv2 rotate codes lazily in connect().
        self._rotation_code = None

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"UvcZedCamera({self.device}) is already connected."
            )

        if self.mock:
            import tests.cameras.mock_cv2 as cv2
        else:
            import cv2

        if self.rotation == 90:
            self._rotation_code = cv2.ROTATE_90_CLOCKWISE
        elif self.rotation == -90:
            self._rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif self.rotation == 180:
            self._rotation_code = cv2.ROTATE_180

        cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise OSError(
                f"Can't open ZED UVC device {self.device} (index {self.device_index}). "
                f"Check it is connected (v4l2-ctl --list-devices) and readable."
            )

        # Request the FULL side-by-side frame: width is 2x the per-eye output.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width * 2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps is not None:
            cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != self.width * 2 or actual_h != self.height:
            cap.release()
            raise OSError(
                f"ZED UVC device {self.device} returned {actual_w}x{actual_h} for the "
                f"side-by-side frame, but expected {self.width * 2}x{self.height} "
                f"(per-eye {self.width}x{self.height}). Adjust --robot.cameras width/height."
            )

        # Warm up: discard the first frames while auto-exposure settles.
        for _ in range(10):
            cap.read()

        self.camera = cap
        self.is_connected = True

    def _crop_eye(self, frame: np.ndarray) -> np.ndarray:
        w = frame.shape[1]
        half = w // 2
        return frame[:, half:] if self.eye == "right" else frame[:, :half]

    def read(self, temporary_color_mode: str | None = None) -> np.ndarray:
        """Grab one frame, crop to the configured eye, return (H, W, 3)."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"UvcZedCamera({self.device}) is not connected. Try running `camera.connect()` first."
            )

        if self.mock:
            import tests.cameras.mock_cv2 as cv2
        else:
            import cv2

        start_time = time.perf_counter()

        ret, frame = self.camera.read()
        if not ret:
            raise OSError(f"Can't capture image from ZED UVC device {self.device}.")

        color_image = self._crop_eye(frame)

        requested_color_mode = self.color_mode if temporary_color_mode is None else temporary_color_mode
        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )
        # OpenCV delivers BGR; LeRobot trains on RGB by default.
        if requested_color_mode == "rgb":
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        h, w, _ = color_image.shape
        if h != self.height or w != self.width:
            raise OSError(
                f"Captured ZED eye image {w}x{h} != expected {self.width}x{self.height}."
            )

        if self._rotation_code is not None:
            color_image = cv2.rotate(color_image, self._rotation_code)

        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
        self.logs["timestamp_utc"] = capture_timestamp_utc()
        self.color_image = color_image
        return color_image

    def read_loop(self):
        while not self.stop_event.is_set():
            try:
                self.color_image = self.read()
            except Exception as e:
                print(f"Error reading in ZED UVC thread: {e}")

    def async_read(self) -> np.ndarray:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"UvcZedCamera({self.device}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        fps = self.fps or 30
        num_tries = 0
        while True:
            if self.color_image is not None:
                return self.color_image
            time.sleep(1 / fps)
            num_tries += 1
            if num_tries > fps * 2:
                raise TimeoutError("Timed out waiting for ZED async_read() to start.")

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"UvcZedCamera({self.device}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
