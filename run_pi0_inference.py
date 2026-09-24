# Run a fine-tuned pi0 model on the xArm6 + gripper MuJoCo scene.
#
# Each loop iteration:
#   1. render the two cameras and read the robot state,
#   2. ask pi0 for an action (given the images, state and TASK text),
#   3. send the action to the robot and step the physics.
#
# How to run (MuJoCo, torch and LeRobot are installed in the "lerobot" conda env only):
#   conda activate lerobot
#   python run_pi0_inference.py

import os
import sys

# Render the camera images for pi0 offscreen with EGL, separately from the viewer window.
# (If both use the viewer's GLFW, Python segfaults when the window is closed.)
# This must be set before mujoco is imported.
os.environ["MUJOCO_GL"] = "egl"

try:
    import mujoco
    import mujoco.viewer
    import numpy as np
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.utils import prepare_observation_for_inference
except ImportError as error:
    sys.exit(f"Missing package: {error.name}\nActivate the lerobot env first:  conda activate lerobot")


# ----------------------------------------------------------------------------
# Settings -- change these
# ----------------------------------------------------------------------------

# Folder of your fine-tuned checkpoint (the one containing config.json and model.safetensors).
MODEL_PATH = "path/to/your/finetuned_pi0/pretrained_model"

# Instruction given to pi0.
TASK = "pick up the red cube"

# How many actions per second pi0 sends. Match the fps of your training data.
CONTROL_HZ = 30


# ----------------------------------------------------------------------------
# Robot details (from scene.xml / xarm6_gripper.xml)
# ----------------------------------------------------------------------------

SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene.xml")

NUM_ARM_JOINTS = 6                    # qpos[0:6] and ctrl[0:6] are the arm joints
GRIPPER_CTRL_INDEX = 6                # ctrl[6] is the gripper: 0 = open, 255 = closed
GRIPPER_CTRL_CLOSED = 255
GRIPPER_JOINT = "left_driver_joint"   # this joint goes from 0 (open) to 0.85 (closed)
GRIPPER_JOINT_CLOSED = 0.85

# pi0's first image input gets the front camera, the second gets the wrist camera.
CAMERAS = ["front_cam", "wrist_cam"]


def load_scene():
    """Load scene.xml, add the two cameras, and put the robot in its home pose."""
    spec = mujoco.MjSpec.from_file(SCENE_PATH)

    # Camera in front of the robot, looking back at it and slightly down.
    spec.worldbody.add_camera(
        name="front_cam",
        pos=[1.3, 0, 0.9],
        xyaxes=[0, 1, 0, -0.5, 0, 1],
    )

    # Camera on the side of the gripper, looking out past the fingers.
    spec.body("link_eef").add_camera(
        name="wrist_cam",
        pos=[0.09, 0, 0.03],
        xyaxes=[0, 1, 0, 0.97, 0, 0.26],
    )

    model = spec.compile()
    data = mujoco.MjData(model)

    # "home" is defined in scene.xml: tool pointing down above the block, gripper open.
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def load_policy():
    """Load pi0 and the pre/post-processors saved with it.

    The preprocessor tokenizes the task text and normalizes the images and state.
    The postprocessor turns the model's normalized action back into real units.
    """
    if not os.path.isdir(MODEL_PATH):
        sys.exit(f"Model folder not found: {MODEL_PATH}\nSet MODEL_PATH at the top of this script.")

    policy = PI0Policy.from_pretrained(MODEL_PATH)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=MODEL_PATH)
    return policy, preprocessor, postprocessor


def get_observation(data, renderer, image_names):
    """Collect what pi0 sees: one image per camera plus the robot state."""
    observation = {}

    for image_name, camera in zip(image_names, CAMERAS):
        renderer.update_scene(data, camera=camera)
        observation[image_name] = renderer.render()

    # State = 6 arm joint angles + how closed the gripper is (0 = open, 1 = closed).
    arm_joints = data.qpos[:NUM_ARM_JOINTS]
    gripper = data.joint(GRIPPER_JOINT).qpos[0] / GRIPPER_JOINT_CLOSED
    observation["observation.state"] = np.append(arm_joints, gripper).astype(np.float32)

    return observation


def predict_action(policy, preprocessor, postprocessor, observation):
    """Ask pi0 for the next action."""
    batch = prepare_observation_for_inference(observation, policy.config.device, task=TASK)
    batch = preprocessor(batch)

    with torch.no_grad():
        action = policy.select_action(batch)

    action = postprocessor(action)
    return action[0].float().numpy()  # first (only) item in the batch, as float32 numpy


def apply_action(data, action):
    """Send the action to the robot.

    action[0:6] = arm joint targets in radians
    action[6]   = gripper command, 0 = open, 1 = closed
    """
    data.ctrl[:NUM_ARM_JOINTS] = action[:NUM_ARM_JOINTS]
    data.ctrl[GRIPPER_CTRL_INDEX] = action[NUM_ARM_JOINTS] * GRIPPER_CTRL_CLOSED


def main():
    model, data = load_scene()
    policy, preprocessor, postprocessor = load_policy()

    image_names = list(policy.config.image_features)
    renderer = mujoco.Renderer(model, height=224, width=224)  # pi0 works on 224x224 images

    # The physics runs much faster than CONTROL_HZ, so step it several times per action.
    physics_steps_per_action = int(1 / (CONTROL_HZ * model.opt.timestep))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            observation = get_observation(data, renderer, image_names)
            action = predict_action(policy, preprocessor, postprocessor, observation)
            apply_action(data, action)

            for _ in range(physics_steps_per_action):
                mujoco.mj_step(model, data)
            viewer.sync()

    renderer.close()


if __name__ == "__main__":
    main()
