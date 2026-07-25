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

import abc
from dataclasses import dataclass, field
from typing import Sequence

import draccus

from lerobot.common.robot_devices.cameras.configs import (
    AzureKinectCameraConfig,
    CameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.motors.configs import (
    DynamixelMotorsBusConfig,
    FeetechMotorsBusConfig,
    MotorsBusConfig,
)


@dataclass
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


# TODO(rcadene, aliberts): remove ManipulatorRobotConfig abstraction
@dataclass
class ManipulatorRobotConfig(RobotConfig):
    leader_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    # Optionally limit the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length
    # as the number of motors in your follower arms (assumes all follower arms have the same number of
    # motors).
    max_relative_target: list[float] | float | None = None

    # Optionally set the leader arm in torque mode with the gripper motor set to this angle. This makes it
    # possible to squeeze the gripper and have it spring back to an open position on its own. If None, the
    # gripper is not put in torque mode.
    gripper_open_degree: float | None = None

    mock: bool = False

    # save end-effector pose info (enabled for aloha)
    use_eef: bool = False

    def __post_init__(self):
        if self.mock:
            for arm in self.leader_arms.values():
                if not arm.mock:
                    arm.mock = True
            for arm in self.follower_arms.values():
                if not arm.mock:
                    arm.mock = True
            for cam in self.cameras.values():
                if not cam.mock:
                    cam.mock = True

        if self.max_relative_target is not None and isinstance(self.max_relative_target, Sequence):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(self.max_relative_target):
                    raise ValueError(
                        f"len(max_relative_target)={len(self.max_relative_target)} but the follower arm with name {name} has "
                        f"{len(self.follower_arms[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )


@RobotConfig.register_subclass("aloha")
@dataclass
class AlohaRobotConfig(ManipulatorRobotConfig):
    # Specific to Aloha, LeRobot comes with default calibration files. Assuming the motors have been
    # properly assembled, no manual calibration step is expected. If you need to run manual calibration,
    # simply update this path to ".cache/calibration/aloha"
    calibration_dir: str = ".cache/calibration/aloha_default"

    # /!\ FOR SAFETY, READ THIS /!\
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    # For Aloha, for every goal position request, motor rotations are capped at 5 degrees by default.
    # When you feel more confident with teleoperation or running the policy, you can extend
    # this safety limit and even removing it by setting it to `null`.
    # Also, everything is expected to work safely out-of-the-box, but we highly advise to
    # first try to teleoperate the grippers only (by commenting out the rest of the motors in this yaml),
    # then to gradually add more motors (by uncommenting), until you can teleoperate both arms fully
    max_relative_target: int | None = 5

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                # window_x
                port="/dev/ttyDXL_leader_left",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm430-w350"],
                    "shoulder": [2, "xm430-w350"],
                    "shoulder_shadow": [3, "xm430-w350"],
                    "elbow": [4, "xm430-w350"],
                    "elbow_shadow": [5, "xm430-w350"],
                    "forearm_roll": [6, "xm430-w350"],
                    "wrist_angle": [7, "xm430-w350"],
                    "wrist_rotate": [8, "xl430-w250"],
                    "gripper": [9, "xc430-w150"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                # window_x
                port="/dev/ttyDXL_leader_right",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm430-w350"],
                    "shoulder": [2, "xm430-w350"],
                    "shoulder_shadow": [3, "xm430-w350"],
                    "elbow": [4, "xm430-w350"],
                    "elbow_shadow": [5, "xm430-w350"],
                    "forearm_roll": [6, "xm430-w350"],
                    "wrist_angle": [7, "xm430-w350"],
                    "wrist_rotate": [8, "xl430-w250"],
                    "gripper": [9, "xc430-w150"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_left",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_right",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),
        }
    )

    # Troubleshooting: If one of your IntelRealSense cameras freeze during
    # data recording due to bandwidth limit, you might need to plug the camera
    # on another USB hub or PCIe card.
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_high": IntelRealSenseCameraConfig(
                serial_number=128422271347,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_low": IntelRealSenseCameraConfig(
                serial_number=130322270656,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_left_wrist": IntelRealSenseCameraConfig(
                serial_number=218622272670,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_right_wrist": IntelRealSenseCameraConfig(
                serial_number=130322272300,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("koch")
@dataclass
class KochRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/koch"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0085511",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # ~ Koch specific settings ~
    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False


@RobotConfig.register_subclass("koch_bimanual")
@dataclass
class KochBimanualRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/koch_bimanual"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0085511",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem575E0031751",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem575E0032081",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # ~ Koch specific settings ~
    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False


@RobotConfig.register_subclass("moss")
@dataclass
class MossRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/moss"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58760431091",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("so101")
@dataclass
class So101RobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/so101"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58760431091",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("so100")
@dataclass
class So100RobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/so100"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58760431091",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("droid")
@dataclass
class DroidRobotConfig(RobotConfig):
    # GELLO leader arm config
    gello_port: str | None = None  # Auto-detected from /dev/serial/by-id/* if None
    gello_joint_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    # HACK: HARDCODED FOR A SPECIFIC GELLO
    gello_joint_offsets: tuple[float, ...] = (
        3 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
        4 * 3.141592653589793 / 2,
        2 * 3.141592653589793 / 2,
        2 * 3.141592653589793 / 2,
        2 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
    )
    gello_joint_signs: tuple[int, ...] = (1, 1, 1, 1, 1, -1, 1)
    gello_gripper_joint_id: int = 8
    gello_gripper_open_degrees: int = 272
    gello_gripper_close_degrees: int = 234

    # Deoxys / Franka config
    deoxys_general_cfg_file: str = "lerobot/common/robot_devices/robots/franka_configs/charmander_droid.yml"
    deoxys_controller_type: str = "JOINT_IMPEDANCE"
    deoxys_controller_cfg_file: str = "lerobot/common/robot_devices/robots/franka_configs/joint-impedance-controller.yml"

    # Teleop mapping: scale + sign for delta mapping from GELLO to Franka
    mapping_coefficients: tuple[float, ...] = (0.8, -0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
    gripper_threshold: float = 0.5
    gripper_open_action: float = 1.0
    gripper_close_action: float = 0.0

    # Robotiq gripper config
    robotiq_port: str | None = None  # Auto-detected by pyRobotiqGripper if None

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_main": AzureKinectCameraConfig(
                device_id=0,
                fps=30,
                width=1280,
                height=720,
            ),
        }
    )

    # Max joint delta (rad) before smooth interpolation kicks in during teleop.
    # Prevents jerky motion when GELLO drifts between episodes.
    max_safe_joint_delta: float = 0.3

    # save end-effector pose info
    use_eef: bool = True
    mock: bool = False


@RobotConfig.register_subclass("franka_leap")
@dataclass
class FrankaLeapRobotConfig(RobotConfig):
    # GELLO leader arm config
    gello_port: str | None = None  # Auto-detected from /dev/serial/by-id/* if None
    gello_joint_ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    # HACK: HARDCODED FOR A SPECIFIC GELLO
    gello_joint_offsets: tuple[float, ...] = (
        4 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
        2 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
        4 * 3.141592653589793 / 2,
        0 * 3.141592653589793 / 2,
    )
    gello_joint_signs: tuple[int, ...] = (1, 1, 1, 1, 1, -1, 1)
    gello_gripper_joint_id: int = 8
    gello_gripper_open_degrees: int = 195
    gello_gripper_close_degrees: int = 152

    # Deoxys / Franka config
    deoxys_general_cfg_file: str = "lerobot/common/robot_devices/robots/franka_configs/charmander_leap.yml"
    deoxys_controller_type: str = "JOINT_IMPEDANCE"
    deoxys_controller_cfg_file: str = "lerobot/common/robot_devices/robots/franka_configs/joint-impedance-controller.yml"

    # Teleop mapping
    max_safe_joint_delta: float = 0.3

    # LEAP hand + Manus glove config
    geort_checkpoint_root: str = "/home/leap/Desktop/GeoRT/checkpoint"
    geort_ckpt_tag: str = "sriram_1"

    # save end-effector pose info
    use_eef: bool = True

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_main": AzureKinectCameraConfig(
                device_id=0,
                fps=30,
                width=1280,
                height=720,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("dexbot_sharpa")
@dataclass
class DexbotSharpaRobotConfig(RobotConfig):
    """Config for recording a dexbot-teleop Sharpa-hand session over ZMQ.

    This robot is a *pure recorder*: the dexbot-teleop stack (in its own env)
    keeps running its publishers, and this class only subscribes to them:
      - ``sensor_<side>``  from sharpa_subscriber.py  (hand joints/torques/tactile)
      - ``wrist_<side>``   from publish_wrist_wrench.py (wrist F/T + measured arm q)
      - ``<hand>_<side>``  (arm) from teleop-retargeting (commanded arm_joint_positions)
      - ``<hand>_<side>``  (hand) from wuji-teleop        (commanded joint_positions)

    In the canonical ``start_teleop.sh`` pipeline the commanded *arm* pose and
    commanded *hand* pose are published by two different processes on two
    different ports (arm: teleop-retargeting @ 5557, hand: wuji-teleop @ 5560),
    so they are configured separately below. For a standalone
    ``teleop.py --franka-ik`` run that publishes both on one port, set
    ``teleop_hand_port`` equal to ``teleop_arm_port``.

    No hardware is opened and no commands are sent; ``send_action`` is a no-op.
    """

    # Which hand/side we are recording.
    side: str = "right"          # "left" | "right"
    hand: str = "sharpa"         # teleop topic prefix ("<hand>_<side>")

    # ZMQ endpoints of the already-running dexbot publishers.
    sensor_host: str = "localhost"
    sensor_port: int = 5561      # sharpa_subscriber --sensor-pub-port
    wrist_host: str = "localhost"
    wrist_port: int = 5563       # publish_wrist_wrench --pub-port
    # Commanded ARM pose source (teleop-retargeting / teleop.py PUB).
    teleop_arm_host: str = "localhost"
    teleop_arm_port: int = 5557
    # Commanded HAND pose source (wuji-teleop PUB). For a single-process teleop
    # that publishes both arm and hand on one port, set this to the arm port.
    teleop_hand_host: str = "localhost"
    teleop_hand_port: int = 5560

    # Live controller-config streams (optional, for airtight provenance). When
    # the dexbot subscribers are run with their config publishers enabled, the
    # recorder subscribes and OVERRIDES the manually-passed metadata below with
    # the actual live values. Hand config rides the sensor port (topic
    # hand_config_<side>); arm config needs its own port (arm_config_<side>).
    # 0 disables (fall back to the manually-passed values only).
    hand_config_from_stream: bool = True   # listen on sensor_port for hand_config
    arm_config_port: int = 0               # tianji-subscriber --config-pub-port
    arm_config_host: str = "localhost"

    # Coordinated-reset PUB: between episodes the recorder publishes "reset_arm"
    # so tianji-subscriber homes + re-tares the arm and teleop re-anchors. 0
    # disables (the subscriber's manual ENTER reset still works).
    reset_pub_port: int = 0                # tianji-subscriber --reset-sub-port
    # SUB port for 'reset_complete' from teleop-retargeting. When set, send_reset()
    # BLOCKS until the whole handshake finishes (arm homed + re-tared, tracker
    # repositioned, Vive→arm re-anchored), so the next episode never starts
    # mid-reset. 0 = fire-and-forget (rely on reset_time_s). See
    # teleop-retargeting --reset-complete-port.
    reset_complete_port: int = 0
    reset_complete_host: str = "localhost"
    reset_timeout_s: float = 60.0          # max wait for reset_complete

    # Seconds to wait for the first frame on each subscribed stream at connect.
    connect_timeout_s: float = 10.0

    # ── Controller-setting metadata ───────────────────────────────────────
    # These knobs live in the dexbot subscriber processes (sharpa-subscriber /
    # tianji-subscriber) and are NOT observable over ZMQ, so pass here whatever
    # you launched those processes with. They are written verbatim into the
    # dataset's meta/info.json under "dexbot_controller" for provenance. Leave
    # a field None if it does not apply (e.g. MIT gains in position mode).
    hand_control_mode: str = "position"          # "position" | "mit"
    hand_mit_kp_scale: float | None = None       # sharpa-subscriber --mit-kp-scale
    hand_mit_kd_scale: float | None = None       # --mit-kd-scale
    hand_filter_alpha: float | None = None       # --filter-alpha (mit mode)
    hand_speed_coeff: float | None = None        # --speed-coeff
    hand_current_coeff: float | None = None       # --current-coeff
    hand_compliance: bool = False                # --compliance
    hand_kz: float | None = None                 # --kz  (compliance vertical stiffness N/m)
    hand_dz: float | None = None                 # --dz  (compliance vertical damping N*s/m)
    hand_kf: float | None = None                 # --kf  (position-mode force->dq gain)
    hand_contact_threshold: float | None = None  # --contact-threshold (N*m)
    hand_orientation: str | None = None          # --hand-orientation
    # Force-cap (admittance) finger mode — separate from hand_compliance.
    hand_force_cap_mode: bool = False            # --force-cap-mode
    hand_force_cap: float | None = None          # --force-cap (N)
    hand_force_cap_ki: float | None = None       # --force-cap-ki (compliance, rad per N)
    hand_force_cap_retreat_lag: float | None = None  # --force-cap-retreat-lag (first-order lag gamma)
    hand_force_cap_release_frac: float | None = None  # --force-cap-release-frac
    hand_force_cap_open_rate: float | None = None    # --force-cap-open-rate (rad/s)
    hand_force_cap_close_rate: float | None = None   # --force-cap-close-rate (rad/s)
    hand_force_cap_limit_margin: float | None = None  # --force-cap-limit-margin (rad)
    hand_force_cap_exclude: str | None = None    # --force-cap-exclude (stiff fingers)
    hand_force_cap_source: str | None = None     # --force-cap-source (torque|tactile)
    arm_type: str | None = None                  # "tianji" | "franka"
    arm_max_delta_deg: float | None = None       # tianji-subscriber --max-delta-deg
    arm_max_lag_deg: float | None = None         # --max-lag-deg
    teleop_pub_hz: float | None = None           # teleop-retargeting --pub-hz
    # Arm wrench-admittance metadata (teleop-retargeting --admittance; the arm
    # yields the EE target under contact force). Not observable over ZMQ, so
    # pass whatever teleop-retargeting was launched with.
    arm_admittance: bool = False                 # --admittance
    arm_admittance_gain: float | None = None     # --admittance-gain (m/s per N)
    arm_admittance_deadband_n: float | None = None  # --admittance-deadband-n (N)
    arm_admittance_max_offset_m: float | None = None  # --admittance-max-offset-m (m)
    arm_admittance_leak_tau_s: float | None = None  # --admittance-leak-tau-s (s)
    arm_admittance_sign: float | None = None     # --admittance-sign
    # Free-form catch-all for anything not covered above (merged into the
    # metadata block). E.g. --robot.controller_extra='{"note":"soft grasp"}'.
    controller_extra: dict = field(default_factory=lambda: {})

    # Recorder has no cameras by default (dexbot teleop loop has none). Add
    # OpenCV/RealSense camera configs here to fold vision into the dataset.
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    use_eef: bool = False
    mock: bool = False


@RobotConfig.register_subclass("dummy")
@dataclass
class DummyRobotConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})
    use_eef: bool = False
    mock: bool = False


@RobotConfig.register_subclass("stretch")
@dataclass
class StretchRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "navigation": OpenCVCameraConfig(
                camera_index="/dev/hello-nav-head-camera",
                fps=10,
                width=1280,
                height=720,
                rotation=-90,
            ),
            "head": IntelRealSenseCameraConfig(
                name="Intel RealSense D435I",
                fps=30,
                width=640,
                height=480,
                rotation=90,
            ),
            "wrist": IntelRealSenseCameraConfig(
                name="Intel RealSense D405",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # Network Configuration
    ip: str = "192.168.0.193"
    port: int = 5555
    video_port: int = 5556

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": OpenCVCameraConfig(
                camera_index="/dev/video0", fps=30, width=640, height=480, rotation=90
            ),
            "wrist": OpenCVCameraConfig(
                camera_index="/dev/video2", fps=30, width=640, height=480, rotation=180
            ),
        }
    )

    calibration_dir: str = ".cache/calibration/lekiwi"

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0077581",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM0",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                    "left_wheel": (7, "sts3215"),
                    "back_wheel": (8, "sts3215"),
                    "right_wheel": (9, "sts3215"),
                },
            ),
        }
    )

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    mock: bool = False
