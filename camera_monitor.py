import cv2
import numpy as np
import time
import threading
import os
from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sanitize_rtsp_url(rtsp_url):
    parsed_url = urlparse(str(rtsp_url))
    safe_path = f"{parsed_url.netloc}{parsed_url.path}".replace('/', '_').replace(':', '_')
    return safe_path

class CameraMonitor:
    def __init__(self, source, camera_id=None):
        self.camera_id = camera_id
        self.source = source
        self.is_rtsp = isinstance(source, str) and source.startswith('rtsp://')
        self.cap = None
        self.fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.alert_delay = 120  # 2 minutes
        self.brightness_threshold = 5  # ±5 from reference
        self.angle_match_threshold = 0.75  # ORB match ratio

        # Initialize fallback frame
        self._update_fallback_frame("Initializing...")

        self._connect_to_stream()
        time.sleep(2)  # Reduced for faster startup

        ret, ref_frame = self.cap.read() if self.cap and self.cap.isOpened() else (False, None)
        if not ret:
            logger.error(f"Camera {self.camera_id}: Failed to capture reference frame from {source}")
            self.ref_frame = self.fallback_frame.copy()
            self.ref_brightness = 0.0  # Default for invalid reference
        else:
            self.ref_frame = ref_frame
            self.ref_brightness = self._get_brightness(ref_frame)

        self.brightness_lower = self.ref_brightness - self.brightness_threshold
        self.brightness_upper = self.ref_brightness + self.brightness_threshold
        self.curr_brightness = self.ref_brightness
        self.lighting_change_detected = False
        self.angle_change_detected = False
        self.lighting_alert_raised = False
        self.angle_alert_raised = False
        self.lighting_change_start = None
        self.angle_change_start = None
        self.lighting_alert_time = None
        self.angle_alert_time = None

        self.latest_frame = self.ref_frame.copy()
        self.frame_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.lock = threading.Lock()
        self.running = True

        safe_folder = sanitize_rtsp_url(str(source)) if self.is_rtsp else f"webcam_{camera_id}"
        self.alert_dir = os.path.join("alert_images", safe_folder)
        os.makedirs(self.alert_dir, exist_ok=True)
        self.log_file = os.path.join(self.alert_dir, "light.log")
        self.last_log_time = time.time()

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _update_fallback_frame(self, message):
        self.fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        text = f"Camera {self.camera_id}: {message} at {ts}"
        cv2.putText(self.fallback_frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def _connect_to_stream(self):
        max_retries = 15
        for attempt in range(max_retries):
            logger.debug(f"Camera {self.camera_id}: Connecting to {self.source} (Attempt {attempt + 1}/{max_retries})")
            backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY] if self.is_rtsp else [cv2.CAP_DSHOW, cv2.CAP_ANY]
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(self.source, backend)
                    if self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        ret, frame = self.cap.read()
                        if ret and frame is not None and frame.size > 0:
                            logger.info(f"Camera {self.camera_id}: Connected with backend {backend}")
                            return
                        self.cap.release()
                    logger.debug(f"Camera {self.camera_id}: Backend {backend} failed")
                except Exception as e:
                    logger.warning(f"Camera {self.camera_id}: Backend {backend} error: {str(e)}")
                if self.cap is not None:
                    self.cap.release()
                time.sleep(0.5)
            time.sleep(2)
        logger.error(f"Camera {self.camera_id}: Failed to connect after {max_retries} attempts")
        self.cap = None
        self._update_fallback_frame("Connection Failed")

    def _get_brightness(self, frame):
        if frame is None or frame.size == 0 or np.array_equal(frame, self.fallback_frame):
            logger.debug(f"Camera {self.camera_id}: Invalid frame for brightness")
            return self.curr_brightness  # Return last known value
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:, :, 2])
            logger.debug(f"Camera {self.camera_id}: Brightness: {brightness:.2f}")
            return brightness
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Brightness calculation error: {str(e)}")
            return self.curr_brightness

    def _detect_angle_change(self, frame1, frame2):
        if frame1 is None or frame2 is None or frame1.size == 0 or frame2.size == 0 or np.array_equal(frame1, self.fallback_frame) or np.array_equal(frame2, self.fallback_frame):
            logger.debug(f"Camera {self.camera_id}: Invalid frame for angle detection")
            return False  # Avoid false positives on invalid frames
        try:
            orb = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(frame1, None)
            kp2, des2 = orb.detectAndCompute(frame2, None)

            if des1 is None or des2 is None:
                logger.debug(f"Camera {self.camera_id}: No descriptors found")
                return True

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if len(matches) < 10:
                logger.debug(f"Camera {self.camera_id}: Insufficient matches ({len(matches)})")
                return True

            good_matches = [m for m in matches if m.distance < 50]
            match_ratio = len(good_matches) / max(len(matches), 1)
            logger.debug(f"Camera {self.camera_id}: Angle match ratio: {match_ratio:.2f}")
            return match_ratio < self.angle_match_threshold
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Angle detection error: {str(e)}")
            return False

    def _capture_loop(self):
        frame_count = 0
        start_time = time.time()
        while self.running:
            if not self.cap or not self.cap.isOpened():
                logger.debug(f"Camera {self.camera_id}: Stream not open. Reconnecting...")
                if self.cap is not None:
                    self.cap.release()
                self._connect_to_stream()
                if not self.cap or not self.cap.isOpened():
                    with self.lock:
                        self._update_fallback_frame("No Signal")
                        self.latest_frame = self.fallback_frame.copy()
                        self.frame_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        self.curr_brightness = 0.0  # Reset brightness
                    time.sleep(1)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                logger.debug(f"Camera {self.camera_id}: Failed to read frame")
                with self.lock:
                    self._update_fallback_frame("Frame Capture Failed")
                    self.latest_frame = self.fallback_frame.copy()
                    self.frame_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    self.curr_brightness = 0.0
                time.sleep(0.1)
                continue

            with self.lock:
                self.latest_frame = frame.copy()
                self.frame_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                frame_count += 1
                curr_brightness = self._get_brightness(frame)
                angle_changed = self._detect_angle_change(self.ref_frame, frame)
                lighting_changed = not (self.brightness_lower <= curr_brightness <= self.brightness_upper) and curr_brightness != 0.0

                self.curr_brightness = curr_brightness
                self.lighting_change_detected = lighting_changed
                self.angle_change_detected = angle_changed

                now = time.time()
                if lighting_changed:
                    if self.lighting_change_start is None:
                        self.lighting_change_start = now
                    elif (now - self.lighting_change_start) >= self.alert_delay:
                        if not self.lighting_alert_raised:
                            self.lighting_alert_raised = True
                            self.lighting_alert_time = time.strftime("%H:%M:%S")
                            self._save_alert_image(frame, "lighting")
                            logger.info(f"Camera {self.camera_id}: Lighting alert raised")
                else:
                    self.lighting_change_start = None
                    self.lighting_alert_raised = False
                    self.lighting_alert_time = None

                if angle_changed:
                    if self.angle_change_start is None:
                        self.angle_change_start = now
                    elif (now - self.angle_change_start) >= self.alert_delay:
                        if not self.angle_alert_raised:
                            self.angle_alert_raised = True
                            self.angle_alert_time = time.strftime("%H:%M:%S")
                            self._save_alert_image(frame, "angle")
                            logger.info(f"Camera {self.camera_id}: Angle alert raised")
                else:
                    self.angle_change_start = None
                    self.angle_alert_raised = False
                    self.angle_alert_time = None

                if now - self.last_log_time >= 15:
                    self._log_brightness()
                    self.last_log_time = now

                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Camera {self.camera_id}: Frame rate: {fps:.2f} FPS")
                    frame_count = 0
                    start_time = time.time()

            time.sleep(0.033)  # ~30 FPS

    def _save_alert_image(self, frame, alert_type):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{alert_type}_alert_{timestamp}.jpg"
        filepath = os.path.join(self.alert_dir, filename)
        try:
            cv2.imwrite(filepath, frame)
            logger.info(f"Camera {self.camera_id}: Saved {alert_type} alert image: {filepath}")
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Failed to save {alert_type} alert image: {str(e)}")

    def _log_brightness(self):
        try:
            with open(self.log_file, "a") as f:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts} brightness: {self.curr_brightness:.2f}\n")
            logger.debug(f"Camera {self.camera_id}: Logged brightness: {self.curr_brightness:.2f}")
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Failed to log brightness: {str(e)}")

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else self.fallback_frame.copy()

    def get_reference_frame(self):
        with self.lock:
            return self.ref_frame.copy() if self.ref_frame is not None else self.fallback_frame.copy()

    def get_frame_timestamp(self):
        with self.lock:
            return self.frame_timestamp

    def set_reference_frame(self):
        with self.lock:
            if self.latest_frame is not None and not np.array_equal(self.latest_frame, self.fallback_frame):
                self.ref_frame = self.latest_frame.copy()
                self.ref_brightness = self._get_brightness(self.ref_frame)
                self.brightness_lower = self.ref_brightness - self.brightness_threshold
                self.brightness_upper = self.ref_brightness + self.brightness_threshold
                self.lighting_change_detected = False
                self.lighting_alert_raised = False
                self.angle_change_detected = False
                self.angle_alert_raised = False
                self.lighting_change_start = None
                self.angle_change_start = None
                self.lighting_alert_time = None
                self.angle_alert_time = None
                logger.info(f"Camera {self.camera_id}: Reference frame updated")
            else:
                logger.warning(f"Camera {self.camera_id}: Cannot set reference frame; using current reference")
                raise ValueError("Invalid frame for reference")

    def get_status(self):
        with self.lock:
            return {
                "camera_id": self.camera_id,
                "brightness": self.curr_brightness,
                "reference_brightness": self.ref_brightness,
                "lighting_change_detected": self.lighting_change_detected,
                "lighting_alert_raised": self.lighting_alert_raised,
                "lighting_alert_time": self.lighting_alert_time,
                "angle_change_detected": self.angle_change_detected,
                "angle_alert_raised": self.angle_alert_raised,
                "angle_alert_time": self.angle_alert_time,
            }

    def stop(self):
        self.running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        logger.info(f"Camera {self.camera_id}: Stopped")