import atexit
import ctypes
import math
import threading
import time

import cv2
from PIL import Image
from typing import List

import scrcpy
from adbutils import adb


class WindowController:
    def __init__(self):
        self.width = None
        self.height = None
        print("Connecting to ADB...")
        try:
            device_list = adb.device_list()
            if not device_list:
                for port in [5555, 16384, 5635] + list(range(5565, 5756, 10)):
                    try:
                        adb.connect(f"127.0.0.1:{port}")
                    except Exception:
                        pass
                device_list = adb.device_list()

            if not device_list:
                raise ConnectionError("No ADB devices found.")

            self.device = device_list[0]
            print(f"Connected to device: {self.device.serial}")

            self.frame_lock = threading.Lock()
            self.scrcpy_client = scrcpy.Client(device=self.device, max_width=0)
            self.last_frame = None
            self.last_joystick_pos = (None, None)

            def on_frame(frame):
                if frame is not None:
                    with self.frame_lock:
                        self.last_frame = frame

            self.scrcpy_client.add_listener(scrcpy.EVENT_FRAME, on_frame)
            self.scrcpy_client.start(threaded=True)
            atexit.register(self.close)
            print("Scrcpy client started successfully.")

        except Exception as e:
            raise ConnectionError(f"Failed to initialize Scrcpy: {e}")

    def get_latest_frame(self):
        """
        Safely retrieves the latest frame.
        Returns None if no frame is available yet.
        """
        with self.frame_lock:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()

    def screenshot(self, array=True):
        frame = self.get_latest_frame()

        while frame is None:
            print("Waiting for first frame...")
            time.sleep(0.1)
            frame = self.get_latest_frame()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not self.width or not self.height:
            self.width = frame.shape[1]
            self.height = frame.shape[0]

        if array:
            return frame_rgb

        return Image.fromarray(frame_rgb)

    def touch_down(self, x, y, pointer_id=1):
        self.scrcpy_client.control.touch(int(x), int(y), scrcpy.ACTION_DOWN, pointer_id)

    def touch_move(self, x, y, pointer_id=1):
        self.scrcpy_client.control.touch(int(x), int(y), scrcpy.ACTION_MOVE, pointer_id)

    def touch_up(self, x, y, pointer_id=1):
        self.scrcpy_client.control.touch(int(x), int(y), scrcpy.ACTION_UP, pointer_id)

    def click(self, x: int, y: int, delay=0.05, touch_up=True, touch_down=True):
        if touch_down: self.touch_down(x, y)
        time.sleep(delay)
        if touch_up: self.touch_up(x, y)

    def close(self):
        if hasattr(self, 'scrcpy_client'):
            self.scrcpy_client.stop()
