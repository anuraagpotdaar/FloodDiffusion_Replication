"""
Streaming joint recovery from 263-dim motion features.
Extracted from FloodDiffusion utils for standalone use (only numpy + torch).
"""

import numpy as np
import torch


def qinv(q):
    assert q.shape[-1] == 4, "q must be a tensor of shape (*, 4)"
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def qrot(q, v):
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


class StreamJointRecovery263:
    """
    Stream version of recover_joint_positions_263 that processes one frame at a time.
    Maintains cumulative state for rotation angles and positions.

    Key insight: The batch version uses PREVIOUS frame's velocity for the current frame,
    so we need to delay the velocity application by one frame.

    Args:
        joints_num: Number of joints in the skeleton
        smoothing_alpha: EMA smoothing factor (0.0 to 1.0)
            - 1.0 = no smoothing (default), output follows input exactly
            - 0.0 = infinite smoothing, output never changes
            - Recommended values: 0.3-0.7 for visible smoothing
            - Formula: smoothed = alpha * current + (1 - alpha) * previous
    """

    def __init__(self, joints_num: int, smoothing_alpha: float = 1.0):
        self.joints_num = joints_num
        self.smoothing_alpha = np.clip(smoothing_alpha, 0.0, 1.0)
        self.reset()

    def reset(self):
        """Reset the accumulated state"""
        self.r_rot_ang_accum = 0.0
        self.r_pos_accum = np.array([0.0, 0.0, 0.0])
        # Store previous frame's velocities for delayed application
        self.prev_rot_vel = 0.0
        self.prev_linear_vel = np.array([0.0, 0.0])
        # Store previous smoothed joints for EMA
        self.prev_smoothed_joints = None

    def process_frame(self, frame_data: np.ndarray) -> np.ndarray:
        """
        Process a single frame and return joint positions for that frame.

        Args:
            frame_data: numpy array of shape (263,) for a single frame

        Returns:
            joints: numpy array of shape (joints_num, 3) representing joint positions
        """
        # Convert to torch tensor
        feature_vec = torch.from_numpy(frame_data).float()

        # Extract current frame's velocities (will be used in NEXT frame)
        curr_rot_vel = feature_vec[0].item()
        curr_linear_vel = feature_vec[1:3].numpy()

        # Update accumulated rotation angle with PREVIOUS frame's velocity FIRST
        # This matches the batch processing: r_rot_ang[i] uses rot_vel[i-1]
        self.r_rot_ang_accum += self.prev_rot_vel

        # Calculate current rotation quaternion using updated accumulated angle
        r_rot_quat = torch.zeros(4)
        r_rot_quat[0] = np.cos(self.r_rot_ang_accum)
        r_rot_quat[2] = np.sin(self.r_rot_ang_accum)

        # Create velocity vector with Y=0 using PREVIOUS frame's velocity
        r_vel = np.array([self.prev_linear_vel[0], 0.0, self.prev_linear_vel[1]])

        # Apply inverse rotation to velocity using CURRENT rotation
        r_vel_torch = torch.from_numpy(r_vel).float()
        r_vel_rotated = qrot(qinv(r_rot_quat).unsqueeze(0), r_vel_torch.unsqueeze(0))
        r_vel_rotated = r_vel_rotated.squeeze(0).numpy()

        # Update accumulated position with rotated velocity
        self.r_pos_accum += r_vel_rotated

        # Get Y position from data
        r_pos = self.r_pos_accum.copy()
        r_pos[1] = feature_vec[3].item()

        # Extract local joint positions
        positions = feature_vec[4 : (self.joints_num - 1) * 3 + 4]
        positions = positions.view(-1, 3)

        # Apply inverse rotation to local joints
        r_rot_quat_expanded = (
            qinv(r_rot_quat).unsqueeze(0).expand(positions.shape[0], 4)
        )
        positions = qrot(r_rot_quat_expanded, positions)

        # Add root XZ to joints
        positions[:, 0] += r_pos[0]
        positions[:, 2] += r_pos[2]

        # Concatenate root and joints
        r_pos_torch = torch.from_numpy(r_pos).float()
        positions = torch.cat([r_pos_torch.unsqueeze(0), positions], dim=0)

        # Convert to numpy
        joints_np = positions.detach().cpu().numpy()

        # Apply EMA smoothing if enabled
        if self.smoothing_alpha < 1.0:
            if self.prev_smoothed_joints is None:
                # First frame, no smoothing possible
                self.prev_smoothed_joints = joints_np.copy()
            else:
                # EMA: smoothed = alpha * current + (1 - alpha) * previous
                joints_np = (
                    self.smoothing_alpha * joints_np
                    + (1.0 - self.smoothing_alpha) * self.prev_smoothed_joints
                )
                self.prev_smoothed_joints = joints_np.copy()

        # Store current velocities for next frame
        self.prev_rot_vel = curr_rot_vel
        self.prev_linear_vel = curr_linear_vel

        return joints_np
