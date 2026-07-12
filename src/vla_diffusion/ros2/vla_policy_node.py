#!/usr/bin/env python3
"""ROS2 node wrapping the trained VLADiffusionPolicy for closed-loop control
on a real (or ROS2-simulated) robot, instead of the LIBERO/robosuite sim
used everywhere else in this project.

This is reference/adapter code: it was written against the standard rclpy
API and a typical Franka-Panda-style ROS2 driver interface (joint_states +
tf2 for end-effector pose, since that's how real Panda ROS2 drivers --
e.g. franka_ros2 -- expose robot state), but it has NOT been run against a
live ROS2 stack. This project's sandbox has no ROS2 install (rclpy isn't a
pip package -- it ships with a ROS2 distro's own Python environment), so
this file is provided as correct, adaptable code rather than a validated
one. See the bottom of this file for what to check on a real ROS2 box.

Topics (rename to match your driver in the launch file, not here):
  Subscribes:
    /agentview/image_raw   sensor_msgs/Image     (rgb8, matches training's camera)
    /joint_states           sensor_msgs/JointState (7 arm joints + 2 gripper joints)
    tf: base_link -> eef_link                    (via tf2, for eef_pos/eef_quat)
  Publishes:
    /vla_policy/action      std_msgs/Float32MultiArray  (7-dim: dx,dy,dz,droll,dpitch,dyaw,gripper)
                             -- deliberately generic; a real deployment remaps
                             this into whatever your robot driver's cartesian
                             velocity + gripper command topics expect.

Parameters:
  checkpoint_path   (string, required)  e.g. outputs/vla_run1/ema.pt
  task_instruction  (string, required)  e.g. "pick up the salad dressing and place it in the basket"
  control_rate_hz    (double, default 20.0)   matches LIBERO's control_freq=20
  exec_horizon        (int, default 4)
  num_inference_steps (int, default 10)
  guidance_scale       (double, default 1.0)
"""
import json
import os

import numpy as np
import rclpy
import tf2_ros
import torch
from diffusers import DDIMScheduler
from rclpy.node import Node
from robosuite.utils.transform_utils import quat2axisangle
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray

from vla_diffusion.data.libero_dataset import ACTION_DIM, PROPRIO_DIM
from vla_diffusion.models.clip_encoders import FrozenClipEncoder
from vla_diffusion.models.vla_diffusion_policy import VLADiffusionPolicy

ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
GRIPPER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
BASE_FRAME = "panda_link0"
EEF_FRAME = "panda_hand"


def denormalize(action_norm, stats):
    action_min = np.array(stats["min"])
    action_max = np.array(stats["max"])
    return (action_norm + 1) / 2 * (action_max - action_min) + action_min


class VLAPolicyNode(Node):
    def __init__(self):
        super().__init__("vla_policy_node")

        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("task_instruction", "")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("exec_horizon", 4)
        self.declare_parameter("num_inference_steps", 10)
        self.declare_parameter("guidance_scale", 1.0)

        checkpoint_path = self.get_parameter("checkpoint_path").value
        if not checkpoint_path:
            raise ValueError("checkpoint_path parameter is required")
        self.task_instruction = self.get_parameter("task_instruction").value
        if not self.task_instruction:
            raise ValueError("task_instruction parameter is required")
        self.exec_horizon = self.get_parameter("exec_horizon").value
        self.num_inference_steps = self.get_parameter("num_inference_steps").value
        self.guidance_scale = self.get_parameter("guidance_scale").value
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt_dir = os.path.dirname(checkpoint_path)
        with open(os.path.join(ckpt_dir, "action_stats.json")) as fp:
            self.action_stats = json.load(fp)
        with open(os.path.join(ckpt_dir, "config.json")) as fp:
            train_config = json.load(fp)
        self.chunk_size = train_config["chunk_size"]

        self.model = VLADiffusionPolicy(proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device).eval()

        self.clip_encoder = FrozenClipEncoder(device=self.device)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=train_config["num_train_timesteps"],
            beta_schedule=train_config["beta_schedule"],
            prediction_type="epsilon",
        )
        with torch.no_grad():
            self.text_embed = self.clip_encoder.encode_text([self.task_instruction]).to(self.device)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_image = None
        self.latest_joint_state = None
        self.action_plan = None
        self.plan_step = 0

        self.image_sub = self.create_subscription(Image, "/agentview/image_raw", self._on_image, 10)
        self.joint_sub = self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.action_pub = self.create_publisher(Float32MultiArray, "/vla_policy/action", 10)

        control_rate_hz = self.get_parameter("control_rate_hz").value
        self.timer = self.create_timer(1.0 / control_rate_hz, self._control_step)

        self.get_logger().info(f"VLA policy node ready. Task: \"{self.task_instruction}\"")

    def _on_image(self, msg: Image):
        # Assumes rgb8 encoding matching training (agentview_rgb, 128x128).
        # A real camera driver may publish a different encoding/resolution --
        # convert/resize here to match what FrozenClipEncoder expects
        # (any square resolution works; it resizes internally to 224x224).
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self.latest_image = arr.copy()

    def _on_joint_state(self, msg: JointState):
        self.latest_joint_state = dict(zip(msg.name, msg.position))

    def _get_proprio(self):
        """Matches vla_diffusion.data.libero_dataset.PROPRIO_KEYS exactly:
        joint_states(7) + gripper_states(2) + ee_states(6, eef_pos + axisangle(eef_quat)).
        """
        if self.latest_joint_state is None:
            return None
        try:
            joint_states = np.array([self.latest_joint_state[n] for n in ARM_JOINT_NAMES])
            gripper_states = np.array([self.latest_joint_state[n] for n in GRIPPER_JOINT_NAMES])
        except KeyError:
            return None  # haven't received a full joint_states message yet

        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, EEF_FRAME, rclpy.time.Time())
        except tf2_ros.TransformException:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        eef_pos = np.array([t.x, t.y, t.z])
        eef_quat = np.array([q.x, q.y, q.z, q.w])
        ee_states = np.concatenate([eef_pos, quat2axisangle(eef_quat)])

        return np.concatenate([joint_states, gripper_states, ee_states]).astype(np.float32)

    def _control_step(self):
        if self.latest_image is None:
            return
        proprio = self._get_proprio()
        if proprio is None:
            self.get_logger().warn("Waiting for joint_states / tf...", throttle_duration_sec=2.0)
            return

        if self.action_plan is None or self.plan_step >= self.exec_horizon:
            image = torch.from_numpy(self.latest_image).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(self.device)

            with torch.no_grad():
                vision_embed = self.clip_encoder.encode_image(image)
                sampled = self.model.sample(
                    vision_embed, self.text_embed, proprio_t, self.chunk_size, self.scheduler,
                    num_inference_steps=self.num_inference_steps, guidance_scale=self.guidance_scale,
                )[0].cpu().numpy()

            self.action_plan = denormalize(sampled, self.action_stats)
            self.plan_step = 0

        action = self.action_plan[self.plan_step]
        self.plan_step += 1

        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)


def main():
    rclpy.init()
    node = VLAPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


# --- Notes for running this on a real ROS2 box (untested here) ---
#
# 1. rclpy, sensor_msgs, std_msgs, tf2_ros come from your ROS2 distro's own
#    Python environment, not pip/uv. This project's deps (torch, diffusers,
#    open_clip, robosuite) need to be importable from *that same*
#    interpreter -- either `pip install` them into the ROS2 venv, or point
#    PYTHONPATH at this project's .venv/lib/python3.x/site-packages.
# 2. ARM_JOINT_NAMES / GRIPPER_JOINT_NAMES / BASE_FRAME / EEF_FRAME are
#    Franka Panda defaults (matching LIBERO's simulated robot) -- rename to
#    match your driver's actual joint/frame names.
# 3. /vla_policy/action publishes the raw 7-dim training action space
#    unchanged. Nothing here maps it to a specific controller's command
#    topic (e.g. cartesian velocity + gripper width) -- that mapping is
#    robot/driver-specific and belongs in a small remapping node or a
#    ros2_control controller, not baked into the policy node.
# 4. Wrap this in a proper ament_python package (setup.py, package.xml,
#    entry_points console_script) before colcon build; this file alone is
#    a runnable node, not a package.
