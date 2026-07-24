"""DexbotSharpaRobot: record a dexbot-teleop Sharpa-hand session as a LeRobot dataset.

This "robot" owns no hardware. The dexbot-teleop stack runs in its own
environment and publishes over ZMQ; this class is a thin SUB client that reads
the latest value on each stream every control tick and assembles a frame:

    sensor_<side>   (sharpa_subscriber.py)     -> hand joints / torques / tactile
    wrist_<side>    (publish_wrist_wrench.py)  -> wrist F/T wrench + measured arm q
    <hand>_<side>   (teleop-retargeting)       -> commanded arm q (arm_joint_positions)
    <hand>_<side>   (wuji-teleop)              -> commanded hand qpos (joint_positions)

In the canonical start_teleop.sh pipeline the arm and hand commands come from
two different processes on two different ports, so they are subscribed
separately. For a single-process teleop.py --franka-ik run set the hand port
equal to the arm port.

Recorded features (all raw, no derived object wrench):

    observation.state          (29,)  measured  arm_q(7) + hand joint_angles(22)
    observation.hand_torque    (22,)  hand joint torques
    observation.tactile        (30,)  fingertip f6 (5 fingers x 6) flattened
    observation.wrist_wrench    (6,)  [Fx,Fy,Fz,Mx,My,Mz]
    action                     (29,)  commanded arm_q_cmd(7) + hand qpos(22)

Because every source is a separate async publisher, this is a "latest-value
join at fps": each tick grabs whatever each SUB socket last received. The loop
cadence (control_loop's busy_wait) sets the nominal sync; per-source skew is
bounded by each publisher's period, not zero. If a stream has not produced a
frame yet, its slot holds zeros (and, for arm/hand commands, the last good
value is carried forward once one arrives).
"""

import json
import threading
import time

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import DexbotSharpaRobotConfig

# Data widths (mirrors dexbot_teleop constants). Kept local so this file does
# not import the dexbot package (separate environment).
NUM_HAND_JOINTS = 22
NUM_ARM_JOINTS = 7
NUM_FINGERS = 5
TACTILE_PER_FINGER = 6
TACTILE_DIM = NUM_FINGERS * TACTILE_PER_FINGER  # 30
WRENCH_DIM = 6

# Human-readable hand joint names, in Sharpa SDK index order.
SHARPA_JOINT_NAMES = [
    "Thumb CMC FE", "Thumb CMC AA", "Thumb MCP FE", "Thumb MCP AA", "Thumb DIP FE",
    "Index MCP FE", "Index MCP AA", "Index PIP FE", "Index DIP FE",
    "Middle MCP FE", "Middle MCP AA", "Middle PIP FE", "Middle DIP FE",
    "Ring MCP FE", "Ring MCP AA", "Ring PIP FE", "Ring DIP FE",
    "Pinky CMC FE", "Pinky MCP FE", "Pinky MCP AA", "Pinky PIP FE", "Pinky DIP FE",
]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


class _ZmqLatest:
    """Background SUB socket keeping the latest JSON message for a topic.

    Reconnect-friendly and O(1) to read. ``got_frame`` flips True once any
    message arrives; ``latest`` returns the last decoded dict (or None).
    """

    def __init__(self, host: str, port: int, topic: str):
        import zmq

        self._latest = None
        self._got = False
        self._lock = threading.Lock()
        self._stop = threading.Event()

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 2)
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        sock.connect(f"tcp://{host}:{port}")
        self._sock = sock
        self.topic = topic

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import zmq

        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            try:
                if not poller.poll(timeout=100):  # ms
                    continue
                self._sock.recv()  # topic frame
                msg = json.loads(self._sock.recv_string())
                with self._lock:
                    self._latest = msg
                    self._got = True
            except Exception:
                time.sleep(0.01)

    @property
    def got_frame(self) -> bool:
        with self._lock:
            return self._got

    @property
    def latest(self):
        with self._lock:
            return self._latest

    def wait_for_first(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.got_frame:
                return True
            time.sleep(0.02)
        return self.got_frame

    def close(self):
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._sock.close(linger=0)
        except Exception:
            pass


class DexbotSharpaRobot:
    """Recorder-only robot bridging dexbot-teleop ZMQ streams into LeRobot."""

    robot_type = "dexbot_sharpa"

    def __init__(self, config: DexbotSharpaRobotConfig | None = None, **kwargs):
        if config is None:
            self.config = DexbotSharpaRobotConfig(**kwargs)
        else:
            self.config = config

        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}
        self.robot_type = self.config.type
        self.use_eef = self.config.use_eef

        self.side = self.config.side
        self.hand = self.config.hand

        # ZMQ sources (created on connect()).
        self._sensor = None
        self._wrist = None
        self._teleop_arm = None
        self._teleop_hand = None
        # Optional live controller-config streams.
        self._hand_cfg = None
        self._arm_cfg = None
        # Optional coordinated-reset sockets (created on connect()).
        self._reset_pub = None            # PUB reset_arm -> tianji-subscriber
        self._reset_complete_sub = None   # SUB reset_complete <- teleop-retargeting

        # Carried-forward last-good command vectors (teleop may briefly drop a
        # field, or arm IK may be inactive at the very start of an episode).
        self._last_arm_q_cmd = np.zeros(NUM_ARM_JOINTS, dtype=np.float32)
        self._last_hand_qpos = np.zeros(NUM_HAND_JOINTS, dtype=np.float32)

        # Feature name lists.
        self.arm_joint_names = [f"arm_joint_{i + 1}" for i in range(NUM_ARM_JOINTS)]
        self.state_names = self.arm_joint_names + list(SHARPA_JOINT_NAMES)
        self.tactile_names = [
            f"{f}_{ax}"
            for f in FINGER_NAMES
            for ax in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
        ]
        self.wrench_names = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    # ── feature schema ────────────────────────────────────────────────────
    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            cam_ft.update(cam.config.get_feature_specs(cam_key))
        return cam_ft

    @property
    def motor_features(self) -> dict:
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(self.state_names),),
                "names": list(self.state_names),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.state_names),),
                "names": list(self.state_names),
            },
            "observation.hand_torque": {
                "dtype": "float32",
                "shape": (NUM_HAND_JOINTS,),
                "names": list(SHARPA_JOINT_NAMES),
            },
            "observation.tactile": {
                "dtype": "float32",
                "shape": (TACTILE_DIM,),
                "names": list(self.tactile_names),
            },
            "observation.wrist_wrench": {
                "dtype": "float32",
                "shape": (WRENCH_DIM,),
                "names": list(self.wrench_names),
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def controller_metadata(self) -> dict:
        """Arm/hand controller settings, for meta/info.json provenance.

        These knobs are not observable over ZMQ (they live in the dexbot
        subscriber processes), so they come from the config you pass at record
        time. ``record()`` writes this under ``info["dexbot_controller"]``.
        """
        c = self.config
        meta = {
            "side": c.side,
            "hand_name": c.hand,
            "sources": {
                "sensor": {"host": c.sensor_host, "port": c.sensor_port,
                           "topic": f"sensor_{c.side}"},
                "wrist_wrench": {"host": c.wrist_host, "port": c.wrist_port,
                                 "topic": f"wrist_{c.side}"},
                "arm_cmd": {"host": c.teleop_arm_host, "port": c.teleop_arm_port,
                            "topic": f"{c.hand}_{c.side}"},
                "hand_cmd": {"host": c.teleop_hand_host, "port": c.teleop_hand_port,
                             "topic": f"{c.hand}_{c.side}"},
            },
            "hand_controller": {
                "control_mode": c.hand_control_mode,
                "mit_kp_scale": c.hand_mit_kp_scale,
                "mit_kd_scale": c.hand_mit_kd_scale,
                "filter_alpha": c.hand_filter_alpha,
                "speed_coeff": c.hand_speed_coeff,
                "current_coeff": c.hand_current_coeff,
                "compliance": c.hand_compliance,
                "kz": c.hand_kz,
                "dz": c.hand_dz,
                "kf": c.hand_kf,
                "contact_threshold": c.hand_contact_threshold,
                "orientation": c.hand_orientation,
                "force_cap_mode": c.hand_force_cap_mode,
                "force_cap": c.hand_force_cap,
                "force_cap_ki": c.hand_force_cap_ki,
                "force_cap_retreat_lag": c.hand_force_cap_retreat_lag,
                "force_cap_release_frac": c.hand_force_cap_release_frac,
                "force_cap_open_rate": c.hand_force_cap_open_rate,
                "force_cap_close_rate": c.hand_force_cap_close_rate,
                "force_cap_limit_margin": c.hand_force_cap_limit_margin,
                "force_cap_exclude": c.hand_force_cap_exclude,
            },
            "arm_controller": {
                "type": c.arm_type,
                "max_delta_deg": c.arm_max_delta_deg,
                "max_lag_deg": c.arm_max_lag_deg,
                "admittance": c.arm_admittance,
                "admittance_gain": c.arm_admittance_gain,
                "admittance_deadband_n": c.arm_admittance_deadband_n,
                "admittance_max_offset_m": c.arm_admittance_max_offset_m,
                "admittance_leak_tau_s": c.arm_admittance_leak_tau_s,
                "admittance_sign": c.arm_admittance_sign,
            },
            "teleop_pub_hz": c.teleop_pub_hz,
        }
        if c.controller_extra:
            meta["extra"] = dict(c.controller_extra)

        # Override with LIVE values from the subscribers' config streams when
        # available — these are ground truth (what the processes actually ran
        # with) and supersede the manually-passed args above.
        meta["hand_controller_source"] = "passed"
        meta["arm_controller_source"] = "passed"
        if self._hand_cfg is not None and self._hand_cfg.latest is not None:
            live = dict(self._hand_cfg.latest)
            live.pop("side", None)
            meta["hand_controller"].update(live)
            meta["hand_controller_source"] = "live_stream"
        if self._arm_cfg is not None and self._arm_cfg.latest is not None:
            live = dict(self._arm_cfg.latest)
            live.pop("side", None)
            meta["arm_controller"].update(live)
            meta["arm_controller_source"] = "live_stream"
        return meta

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    # ── lifecycle ─────────────────────────────────────────────────────────
    def connect(self):
        if self.is_connected:
            raise RuntimeError("DexbotSharpaRobot is already connected.")

        c = self.config
        topic = f"{self.hand}_{self.side}"
        self._sensor = _ZmqLatest(c.sensor_host, c.sensor_port, f"sensor_{self.side}")
        self._wrist = _ZmqLatest(c.wrist_host, c.wrist_port, f"wrist_{self.side}")
        self._teleop_arm = _ZmqLatest(c.teleop_arm_host, c.teleop_arm_port, topic)
        # If arm and hand commands share one publisher/port, reuse the socket.
        same = (c.teleop_hand_host == c.teleop_arm_host
                and c.teleop_hand_port == c.teleop_arm_port)
        self._teleop_hand = (
            self._teleop_arm if same
            else _ZmqLatest(c.teleop_hand_host, c.teleop_hand_port, topic)
        )

        sources = [
            (self._sensor, f"sensor_{self.side} (sharpa_subscriber, port {c.sensor_port})"),
            (self._wrist, f"wrist_{self.side} (publish_wrist_wrench, port {c.wrist_port})"),
            (self._teleop_arm, f"{topic} arm cmd (teleop-retargeting, port {c.teleop_arm_port})"),
        ]
        if not same:
            sources.append(
                (self._teleop_hand, f"{topic} hand cmd (wuji-teleop, port {c.teleop_hand_port})")
            )
        for src, name in sources:
            if src.wait_for_first(c.connect_timeout_s):
                print(f"[DexbotSharpaRobot] connected: {name}")
            else:
                print(
                    f"[DexbotSharpaRobot] WARNING: no data yet on {name}. "
                    f"Recording will use zeros until it arrives."
                )

        # Optional live controller-config streams (best-effort, non-blocking):
        # override the manually-passed metadata with the subscribers' actual
        # settings when their config publishers are enabled.
        if c.hand_config_from_stream:
            self._hand_cfg = _ZmqLatest(c.sensor_host, c.sensor_port, f"hand_config_{self.side}")
        if c.arm_config_port > 0:
            self._arm_cfg = _ZmqLatest(c.arm_config_host, c.arm_config_port, f"arm_config_{self.side}")
        # Give the low-rate (~0.5 Hz) config heartbeats a moment to arrive so
        # the metadata written at record start reflects the live values.
        for src, label in ((self._hand_cfg, "hand_config"), (self._arm_cfg, "arm_config")):
            if src is not None:
                if src.wait_for_first(3.0):
                    print(f"[DexbotSharpaRobot] live {label} stream connected")
                else:
                    print(f"[DexbotSharpaRobot] no live {label} stream; using passed values")

        # Coordinated-reset PUB: bind so tianji-subscriber (--reset-sub-port) can
        # SUB and home + re-tare the arm between episodes.
        if c.reset_pub_port > 0:
            import zmq
            ctx = zmq.Context.instance()
            self._reset_pub = ctx.socket(zmq.PUB)
            self._reset_pub.setsockopt(zmq.SNDHWM, 2)
            self._reset_pub.bind(f"tcp://*:{c.reset_pub_port}")
            print(f"[DexbotSharpaRobot] reset PUB bound on tcp://*:{c.reset_pub_port} "
                  f"(topic: reset_arm)")
            # Connect the completion SUB now (before any reset is sent) so we
            # never miss the 'reset_complete' reply due to slow-joiner drops.
            if c.reset_complete_port > 0:
                self._reset_complete_sub = ctx.socket(zmq.SUB)
                self._reset_complete_sub.setsockopt(zmq.RCVHWM, 4)
                self._reset_complete_sub.setsockopt_string(zmq.SUBSCRIBE, "reset_complete")
                self._reset_complete_sub.connect(
                    f"tcp://{c.reset_complete_host}:{c.reset_complete_port}"
                )
                print(f"[DexbotSharpaRobot] reset-complete SUB on "
                      f"tcp://{c.reset_complete_host}:{c.reset_complete_port}")

        for cam in self.cameras.values():
            cam.connect()

        self.is_connected = True

    def send_reset(self) -> None:
        """Publish 'reset_arm' so tianji-subscriber homes + re-tares the arm and
        teleop re-anchors. No-op if reset_pub_port was not configured.

        If a reset-complete SUB is configured, this BLOCKS until teleop reports
        the full handshake is done (arm homed + re-tared, tracker repositioned,
        Vive→arm re-anchored), so the next episode never starts mid-reset. The
        wait is bounded by ``reset_timeout_s``; on timeout it warns and returns.

        Called by the recorder between episodes (the environment-reset phase).
        """
        if self._reset_pub is None:
            return
        import zmq
        c = self.config
        try:
            # Drop any stale completion from a previous reset so we only ever
            # wait on the reply to THIS request.
            if self._reset_complete_sub is not None:
                while True:
                    try:
                        self._reset_complete_sub.recv_string(zmq.NOBLOCK)
                        self._reset_complete_sub.recv_string(zmq.NOBLOCK)
                    except zmq.Again:
                        break
            self._reset_pub.send_string("reset_arm", zmq.SNDMORE | zmq.NOBLOCK)
            self._reset_pub.send_string(
                json.dumps({"side": self.side, "cmd": "reset"}), zmq.NOBLOCK
            )
            print("[DexbotSharpaRobot] sent reset_arm (home + re-tare + re-anchor).")
        except Exception as e:
            print(f"[DexbotSharpaRobot] send_reset failed: {e}")
            return

        if self._reset_complete_sub is None:
            return
        # Block until teleop signals completion (operator repositioned tracker
        # and teleop re-anchored), or timeout.
        print(f"[DexbotSharpaRobot] waiting for reset_complete "
              f"(timeout {c.reset_timeout_s:.0f}s)...")
        deadline = time.time() + c.reset_timeout_s
        while time.time() < deadline:
            try:
                if self._reset_complete_sub.poll(timeout=200):  # ms
                    self._reset_complete_sub.recv_string()      # topic
                    self._reset_complete_sub.recv_string()      # payload
                    print("[DexbotSharpaRobot] reset_complete received — resuming.")
                    return
            except Exception:
                break
        print("[DexbotSharpaRobot] WARNING: reset_complete not received before "
              "timeout; continuing anyway (arm may not be re-anchored).")

    def run_calibration(self):
        # Recorder-only: nothing to calibrate. The dexbot stack owns calibration.
        pass

    # ── reading ───────────────────────────────────────────────────────────
    def _read_observation(self) -> dict:
        """Assemble the observation dict from the latest value on each stream."""
        # Hand joints / torques / tactile from sensor_<side>.
        arm_q = np.zeros(NUM_ARM_JOINTS, dtype=np.float32)
        hand_q = np.zeros(NUM_HAND_JOINTS, dtype=np.float32)
        hand_tau = np.zeros(NUM_HAND_JOINTS, dtype=np.float32)
        tactile = np.zeros(TACTILE_DIM, dtype=np.float32)
        wrist = np.zeros(WRENCH_DIM, dtype=np.float32)

        s = self._sensor.latest
        if s is not None:
            hand_q = np.asarray(s.get("joint_angles", hand_q), dtype=np.float32)
            hand_tau = np.asarray(s.get("joint_torques", hand_tau), dtype=np.float32)
            f6 = s.get("f6_tactile")
            if f6 is not None:
                tactile = np.asarray(f6, dtype=np.float32).reshape(-1)[:TACTILE_DIM]

        # Wrist wrench + measured arm q from wrist_<side>.
        w = self._wrist.latest
        if w is not None:
            force = w.get("force", [0.0] * 3)
            torque = w.get("torque", [0.0] * 3)
            wrist = np.asarray(list(force) + list(torque), dtype=np.float32)[:WRENCH_DIM]
            q_arm = w.get("q")
            if q_arm is not None:
                arm_q = np.asarray(q_arm, dtype=np.float32)[:NUM_ARM_JOINTS]

        state = np.concatenate([arm_q, hand_q]).astype(np.float32)

        obs = {
            "observation.state": torch.from_numpy(state),
            "observation.hand_torque": torch.from_numpy(hand_tau),
            "observation.tactile": torch.from_numpy(tactile),
            "observation.wrist_wrench": torch.from_numpy(wrist),
        }

        for name, cam in self.cameras.items():
            img = cam.async_read()
            obs[f"observation.images.{name}"] = torch.from_numpy(img)
        return obs

    def _read_action(self) -> dict:
        """Commanded arm pose (arm publisher) + hand pose (hand publisher)."""
        arm_cmd = self._last_arm_q_cmd
        hand_cmd = self._last_hand_qpos

        ta = self._teleop_arm.latest
        if ta is not None:
            a = ta.get("arm_joint_positions")
            if a is not None:
                arm_cmd = np.asarray(a, dtype=np.float32)[:NUM_ARM_JOINTS]
                self._last_arm_q_cmd = arm_cmd

        th = self._teleop_hand.latest
        if th is not None:
            h = th.get("joint_positions")
            if h is not None:
                hand_cmd = np.asarray(h, dtype=np.float32)[:NUM_HAND_JOINTS]
                self._last_hand_qpos = hand_cmd

        action = np.concatenate([arm_cmd, hand_cmd]).astype(np.float32)
        return {"action": torch.from_numpy(action)}

    def teleop_step(self, record_data=False):
        if not self.is_connected:
            raise RuntimeError("DexbotSharpaRobot is not connected. Run `connect()` first.")
        # The dexbot stack already performs teleoperation; we only observe.
        if not record_data:
            return
        observation = self._read_observation()
        action = self._read_action()
        return observation, action

    def capture_observation(self) -> dict:
        if not self.is_connected:
            raise RuntimeError("DexbotSharpaRobot is not connected. Run `connect()` first.")
        return self._read_observation()

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        # Recorder-only: never command the robot. Returned unchanged so replay
        # / policy-eval paths that expect an echo keep working.
        return action

    def print_logs(self):
        pass

    def disconnect(self):
        if not self.is_connected:
            return
        # De-dup in case arm and hand share one socket.
        for src in {id(s): s for s in (
            self._sensor, self._wrist, self._teleop_arm, self._teleop_hand,
            self._hand_cfg, self._arm_cfg,
        ) if s is not None}.values():
            src.close()
        self._sensor = self._wrist = self._teleop_arm = self._teleop_hand = None
        self._hand_cfg = self._arm_cfg = None
        for _sock_attr in ("_reset_pub", "_reset_complete_sub"):
            _sock = getattr(self, _sock_attr, None)
            if _sock is not None:
                try:
                    _sock.close(linger=0)
                except Exception:
                    pass
                setattr(self, _sock_attr, None)
        for cam in self.cameras.values():
            cam.disconnect()
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
