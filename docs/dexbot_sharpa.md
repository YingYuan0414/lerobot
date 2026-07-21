# Dexbot Sharpa (ZMQ recorder)

`dexbot_sharpa` records a **dexbot-teleop** Sharpa-hand teleoperation session into
the LeRobot dataset format. Unlike the other robots in this repo, it owns **no
hardware and sends no commands** — it is a pure recorder.

The dexbot-teleop stack (a separate Python environment) keeps running its normal
teleop + sensor processes, which publish over ZMQ. This robot subscribes to those
streams, samples the latest value on each every control tick, and writes frames
via the standard `control_robot.py record` machinery (episode management, warmup,
reset, keyboard re-record, HF format).

## Why a ZMQ bridge

`dexbot-teleop` runs on its own venv (Python 3.13) and cannot be imported here.
The two processes talk over ZMQ, which is already how the dexbot stack is wired.
Keep the dexbot processes running exactly as usual; this robot only listens.

## Subscribed streams

| Topic | Publisher (dexbot-teleop) | Default port | Fields consumed |
|-------|---------------------------|--------------|-----------------|
| `sensor_<side>` | `sharpa-subscriber` (`--sensor-pub-port`) | 5561 | `joint_angles` (22), `joint_torques` (22), `f6_tactile` (5×6) |
| `wrist_<side>` | `tianji-subscriber` (`--wrench-pub-port`) | 5563 | `force` (3) + `torque` (3), `q` (measured arm, 7) |
| `<hand>_<side>` (arm) | `teleop-retargeting` (`--pub-port`) | 5557 | `arm_joint_positions` (7) |
| `<hand>_<side>` (hand) | `wuji-teleop` (`--pub-port`) | 5560 | `joint_positions` (hand, 22) |

**Split command sources.** In the canonical `start_teleop.sh` pipeline the
commanded *arm* pose and commanded *hand* pose come from two different
processes on two different ports: the arm from `teleop-retargeting` (which runs
with `--hand-pub-port 0`, publishing arm only) and the hand from `wuji-teleop`.
The recorder subscribes to both (`--robot.teleop_arm_port` /
`--robot.teleop_hand_port`). For a single-process `teleop.py --franka-ik` run
that publishes both on one port, set them equal — the socket is then shared.

If a stream has not produced a frame yet, its slot is zeros. For the commanded
arm/hand vectors, the last good value is carried forward once one arrives.

## Prerequisites — processes to run alongside recording

Start these on the dexbot-teleop side (its own env) **before** launching
`control_robot.py`. Easiest is `./start_teleop.sh` (panes T1–T4), plus the
wrist-wrench publisher which it does **not** launch:

| # | Process | Role | Must-have flag for recording |
|---|---------|------|------------------------------|
| T1 | `wuji-teleop --pub-port 5560` | glove → commanded **hand** qpos | publishes `joint_positions` → `action` hand dims |
| T2 | `sharpa-subscriber --port 5560 --sensor-pub-port 5561` | drives hand; publishes sensor stream | **`--sensor-pub-port 5561`** (off by default) → `observation.*` |
| T3 | `tianji-subscriber --wrench-pub-port 5563` | drives the **arm** (homes on connect); ALSO publishes wrist F/T + measured arm q | `wrist_<side>` → `observation.wrist_wrench` + measured arm in `observation.state` |
| T4 | `teleop-retargeting --tianji-ik --hand-pub-port 0 --pub-port 5557` | Vive → commanded **arm** q | publishes `arm_joint_positions` → `action` arm dims |

**One arm client only.** The TJ Marvin controller accepts a *single* UDP client
at a time, so the wrist F/T sensor cannot be read by a separate
`publish_wrist_wrench.py` process while `tianji-subscriber` drives the arm —
starting both makes the second fail with *"port may be occupied"*. Instead
`tianji-subscriber --wrench-pub-port 5563` reads the wrench from its own arm
connection (passive `robot.observe()`; the arm is never additionally commanded
for this) and publishes `wrist_<side>` itself. Run `publish_wrist_wrench.py`
**only** when the arm is not driven (e.g. a wrench-only / manually-posed
session).

Notes:
- Without T4's arm IK active, `arm_joint_positions` is never published and the
  arm dims of `action` stay zero (hand still records).
- Without T3's `--wrench-pub-port` (or a standalone `publish_wrist_wrench.py`
  when the arm is idle), `observation.wrist_wrench` is zeros and the measured
  arm portion of `observation.state` is zeros.
- `tianji-subscriber` needs a `wrench_calib.yaml` per side to publish the
  wrench (see `--wrench-calib`); if missing it warns and skips the wrench but
  keeps driving the arm.
- `start_teleop.sh` uses `HAND_PORT=5560` (wuji PUB == sharpa SUB) and
  `ARM_PORT=5557`; match the recorder's ports to whatever you launched.

## Recording Episodes

```bash
python -m lerobot.scripts.control_robot --robot.type=dexbot_sharpa --control.type=record \
    --control.single_task="Grasp the object." \
    --control.repo_id=<your_hf_user>/dexbot_sharpa_grasp \
    --control.num_episodes=10 \
    --robot.side=right \
    --robot.teleop_arm_port=5557 --robot.teleop_hand_port=5560 \
    --robot.sensor_port=5561 --robot.wrist_port=5563 \
    --control.push_to_hub=false \
    --control.fps=30 --control.reset_time_s=5 --control.warmup_time_s=3 \
    --control.display_data=true
```

Override hosts too if the dexbot publishers run on another machine, e.g.
`--robot.sensor_host=<ip> --robot.teleop_arm_host=<ip>`.

## Recording controller settings (metadata)

The arm/hand controller knobs (MIT gains, compliance stiffness, arm safety
limits, …) live inside the dexbot subscriber processes and are **not** visible
on any ZMQ stream, so the recorder cannot observe them. Pass whatever you
launched those processes with, and they are written verbatim into the dataset's
`meta/info.json` under `dexbot_controller` for provenance:

```bash
    --robot.hand_control_mode=mit \
    --robot.hand_mit_kp_scale=0.5 --robot.hand_mit_kd_scale=0.5 \
    --robot.hand_filter_alpha=0.4 \
    --robot.hand_compliance=true --robot.hand_kz=50 --robot.hand_dz=5 \
    --robot.hand_contact_threshold=0.3 --robot.hand_orientation=palm_down \
    --robot.arm_type=tianji --robot.arm_max_delta_deg=2 --robot.arm_max_lag_deg=10 \
    --robot.teleop_pub_hz=90 \
    --robot.controller_extra='{"note":"soft grasp, half stiffness"}'
```

These fields are optional (unset → `null` in the metadata). `controller_extra`
is a free-form dict merged in for anything not covered by a named field. The
block also records the resolved ZMQ sources (host/port/topic per stream), so a
dataset carries an exact description of how it was produced.

### Live controller-config streams (ground truth)

Manually-passed values can drift from what the subscribers actually ran with.
To make the metadata airtight, the dexbot subscribers can broadcast their
active settings on a low-rate heartbeat (~0.5 Hz, built once at startup — no
per-frame cost, so publishing speed is unaffected):

- `sharpa-subscriber --sensor-pub-port 5561` also publishes `hand_config_<side>`.
- `tianji-subscriber --config-pub-port 5599` publishes `arm_config_<side>`.

When the recorder sees these streams it **overrides** the passed values with the
live ones and records `hand_controller_source` / `arm_controller_source` =
`"live_stream"` (else `"passed"`). Enable on the recorder with
`--robot.arm_config_port=5599` (hand config rides the sensor port automatically;
disable with `--robot.hand_config_from_stream=false`). The `collect_dataset.sh`
launcher wires all of this up by default.

To fold in cameras (they run in *this* env, not dexbot), pass `--robot.cameras='{...}'`
exactly as in [franka_leap.md](franka_leap.md); each camera adds an
`observation.images.<name>` feature.

## State and Action Space

| Feature | Dim | Contents |
|---------|-----|----------|
| `observation.state` | 29 | measured arm q (7) + hand joint angles (22) |
| `observation.hand_torque` | 22 | hand joint torques |
| `observation.tactile` | 30 | fingertip F/T, 5 fingers × [Fx,Fy,Fz,Mx,My,Mz] |
| `observation.wrist_wrench` | 6 | wrist F/T `[Fx,Fy,Fz,Mx,My,Mz]` |
| `action` | 29 | commanded arm q_cmd (7) + commanded hand qpos (22) |

The estimated object wrench (`w_obj`) is **not** recorded — this dataset holds
raw data only. It can be recomputed offline from `observation.state` /
`observation.tactile` / `observation.hand_torque` with dexbot-teleop's
`estimate_object_wrench.py --mode recompute`.

## Synchronization

Each source is an independent async publisher, so a frame is a *latest-value
join* at the recording fps: every tick grabs whatever each SUB socket last
received. The loop cadence (`busy_wait(1/fps)`) sets the nominal alignment;
per-source skew is bounded by each publisher's period, not zero. The dataset
stores a single `timestamp = frame_index / fps` per frame (standard LeRobot
behavior), so it does not capture per-source wall-clock times.
