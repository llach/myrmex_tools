import os
import pickle

sequences = []
datadir   = f"{os.environ['HOME']}/cloth/sequences1/"

# load "edge" samples from the cloth dataset
for fname in os.listdir(datadir):
    if fname in ["pics", ".DS_Store"]: continue # skip directory containing visulizations and macOS bloat
    with open(f"{datadir}{fname}", "rb") as f: 
        """
        each sequence data is of format:

        sequence {
            mm_left {
                timestamp: left_tactile_image # shape (1,16,16)
                ...
            },
            mm_right {
                timestamp: left_tactile_image # shape (1,16,16)
                ...
            },
            joint_states { # gripper finger joint positions (in meters)
                timestamp: {
                    gripper_right_finger_joint: float
                    gripper_left_finger_joint:  float
                },
                ..
            },
            ft {
                timestamp: force-torque measurement # shape (6,)
                ...
            },
            gripper_pose {
                timestamp: (t, q) # gripper pose in base frame coordinates. t is translation (3,), q the rotation as quaternion (4,)
                ...
            }
        }
        """
        sequences.append(pickle.load(f))