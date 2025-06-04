import streamlit as st
import cv2
import time
import logging
from camera_monitor import CameraMonitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="Camera Monitoring Dashboard", layout="wide", initial_sidebar_state="expanded")

# Inject Tailwind CSS via CDN
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .alert-container { transition: all 0.3s ease; }
        .alert-container:hover { transform: scale(1.02); }
        .camera-card { border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .metric-container { background-color: #f3f4f6; padding: 0.5rem; border-radius: 0.25rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="text-3xl font-bold text-gray-800 mb-6">📷 Camera Monitoring Dashboard</h1>', unsafe_allow_html=True)

# Initialize session state
if "monitors" not in st.session_state:
    st.session_state.monitors = {}
if "last_update" not in st.session_state:
    st.session_state.last_update = 0

# Sidebar: Camera sources and controls
with st.sidebar:
    st.markdown('<h2 class="text-xl font-semibold text-gray-700 mb-4">Camera Configuration</h2>', unsafe_allow_html=True)
    camera_sources_input = st.text_area(
        "Enter RTSP/Webcam sources (comma-separated)",
        value="rtsp://admin:Eternal%2412@192.168.1.69:554/Streaming/channels/101",
        help="e.g., rtsp://user:pass@ip:port/stream, 0 for webcam"
    )
    refresh_rate = st.slider(
        "Live View Refresh Rate (seconds)",
        0.033, 1.0, 0.033, 0.01,
        help="Lower values increase frame rate (0.033 ≈ 30 FPS)"
    )
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start Monitoring", key="start")
    with col2:
        stop_button = st.button("Stop Monitoring", key="stop")

# Start cameras
if start_button:
    sources = [src.strip() for src in camera_sources_input.split(",")]
    for cam_id, src in enumerate(sources):
        if src.isdigit():
            src = int(src)
        try:
            if cam_id not in st.session_state.monitors:
                st.session_state.monitors[cam_id] = CameraMonitor(src, camera_id=cam_id)
                st.success(f"Camera {cam_id} started successfully")
                logger.info(f"Started Camera {cam_id} with source: {src}")
        except Exception as e:
            st.error(f"❌ Failed to start Camera {cam_id} ({src}): {str(e)}")
            logger.error(f"Failed to start Camera {cam_id}: {str(e)}")

# Stop cameras
if stop_button:
    for monitor in st.session_state.monitors.values():
        try:
            monitor.stop()
        except Exception as e:
            logger.error(f"Error stopping Camera {monitor.camera_id}: {str(e)}")
    st.session_state.monitors.clear()
    st.success("All cameras stopped")
    logger.info("All cameras stopped")

# Display cameras
if st.session_state.monitors:
    for cam_id, monitor in st.session_state.monitors.items():
        st.markdown(f'<div class="camera-card p-6 mb-6 bg-white">', unsafe_allow_html=True)
        st.markdown(f'<h2 class="text-2xl font-semibold text-gray-700 mb-4">📡 Camera {cam_id}</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown('<h3 class="text-lg font-medium text-gray-600">Live View</h3>', unsafe_allow_html=True)
            frame_placeholder = st.empty()
            timestamp_placeholder = st.empty()
            try:
                frame = monitor.get_latest_frame()
                if frame is None or frame.size == 0:
                    st.warning(f"Camera {cam_id}: No valid frame available")
                    logger.warning(f"Camera {cam_id}: Invalid frame in live view")
                else:
                    frame_placeholder.image(frame, channels="BGR", use_column_width=True)
                    timestamp_placeholder.markdown(
                        f'<p class="text-sm text-gray-500">Frame Timestamp: {monitor.get_frame_timestamp()}</p>',
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error displaying frame for Camera {cam_id}: {str(e)}")
                logger.error(f"Error displaying frame for Camera {cam_id}: {str(e)}")

        with col2:
            st.markdown('<h3 class="text-lg font-medium text-gray-600">Reference Image</h3>', unsafe_allow_html=True)
            try:
                ref_frame = monitor.get_reference_frame()
                if ref_frame is None or ref_frame.size == 0:
                    st.warning(f"Camera {cam_id}: No valid reference frame")
                else:
                    st.image(ref_frame, channels="BGR", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying reference for Camera {cam_id}: {str(e)}")
                logger.error(f"Error displaying reference for Camera {cam_id}: {str(e)}")

            if st.button("Set Reference Frame", key=f"ref_{cam_id}"):
                try:
                    monitor.set_reference_frame()
                    st.success(f"Reference frame updated for Camera {cam_id}")
                    logger.info(f"Reference frame set for Camera {cam_id}")
                except Exception as e:
                    st.error(f"Failed to set reference frame: {str(e)}")
                    logger.error(f"Failed to set reference frame for Camera {cam_id}: {str(e)}")

            try:
                status = monitor.get_status()
                # Ensure brightness metrics are always displayed
                current_brightness = status.get("brightness", 0.0)
                ref_brightness = status.get("reference_brightness", 0.0)
                st.markdown(
                    f'<div class="metric-container mt-4"><p class="text-sm font-medium text-gray-600">Current Brightness: <span class="font-bold">{current_brightness:.2f}</span></p></div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="metric-container mb-4"><p class="text-sm font-medium text-gray-600">Reference Brightness: <span class="font-bold">{ref_brightness:.2f}</span></p></div>',
                    unsafe_allow_html=True
                )

                # Display alerts
                if status.get("lighting_alert_raised", False):
                    st.markdown(
                        f'<div class="alert-container p-3 bg-red-100 text-red-800 rounded-md mb-2">⚠️ Lighting alert! (Since {status.get("lighting_alert_time", "N/A")})</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="p-3 bg-green-100 text-green-800 rounded-md mb-2">✅ No lighting issues</div>',
                        unsafe_allow_html=True
                    )
                if status.get("angle_alert_raised", False):
                    st.markdown(
                        f'<div class="alert-container p-3 bg-red-100 text-red-800 rounded-md">⚠️ Angle alert! (Since {status.get("angle_alert_time", "N/A")})</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="p-3 bg-green-100 text-green-800 rounded-md">✅ No angle issues</div>',
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error retrieving status for Camera {cam_id}: {str(e)}")
                logger.error(f"Error retrieving status for Camera {cam_id}: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)

# Auto-refresh for live view
if st.session_state.monitors:
    current_time = time.time()
    if current_time - st.session_state.last_update >= refresh_rate:
        st.session_state.last_update = current_time
        st.rerun()