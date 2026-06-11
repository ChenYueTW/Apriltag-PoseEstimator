import numpy as np
import math


class PoseEstimator:
    def __init__(self, apriltag_side_length):
        self.camera_pose = np.array([0.088957, 0.030321, 0.070595])
        self.camera_pitch = math.radians(-25.0) # degree
        self.central_sight = self.rotation_matrix_from_axis_angle(np.array([0, 1, 0]), self.camera_pitch) @ np.array([1.0, 0.0, 0.0])
        self.camera_x_axis = np.cross(self.central_sight, np.array([0.0, 0.0, 1.0]))
        self.camera_y_axis = np.cross(self.camera_x_axis, self.central_sight)
        self.apriltag_side_length = apriltag_side_length

    def getPose(self, tx, ty, height):
        target = self.getTargetVector(tx, ty)

        t = (height - self.camera_pose[2]) / target[2]
        apriltag_pose = target * t + self.camera_pose

        return apriltag_pose

    def getTargetVector(self, tx, ty):
        target_x = self.rotation_matrix_from_axis_angle(self.camera_y_axis, math.radians(-tx)) @ self.central_sight
        target_y = self.rotation_matrix_from_axis_angle(self.camera_x_axis, math.radians(ty)) @ self.central_sight
        target = np.cross(np.cross(self.camera_x_axis, target_y), np.cross(self.camera_y_axis, target_x))

        return -self.normalize(target)

    def getCornerPoses(self, center_tx, center_ty, corner_txs_tys):
        if len(corner_txs_tys) != 4:
            raise ValueError("corner_txs_tys must contain exactly 4 corners")

        center_vector = self.getTargetVector(center_tx, center_ty)
        corner_vectors = np.array(
            [self.getTargetVector(tx, ty) for tx, ty in corner_txs_tys],
            dtype=float
        )

        relative_scales = self.getRelativeScales(corner_vectors)
        scaled_corner_vectors = corner_vectors * relative_scales[:, np.newaxis]
        unscaled_edge_lengths = self.getUnscaledEdgeLengths(scaled_corner_vectors)
        average_unscaled_edge_length = np.mean(unscaled_edge_lengths)

        scale = self.apriltag_side_length / average_unscaled_edge_length

        corner_poses = self.camera_pose + scaled_corner_vectors * scale
        center_pose = np.mean(corner_poses, axis=0)
        center_pose_from_center_vector = self.camera_pose + center_vector * np.linalg.norm(
            center_pose - self.camera_pose
        )

        return {
            "center_pose": center_pose,
            "center_pose_from_center_vector": center_pose_from_center_vector,
            "corner_poses": corner_poses,
            "center_vector": center_vector,
            "corner_vectors": corner_vectors,
            "relative_scales": relative_scales,
            "unscaled_edge_lengths": unscaled_edge_lengths,
            "average_unscaled_edge_length": average_unscaled_edge_length,
            "scale": scale,
        }

    def getRelativeScales(self, corner_vectors):
        v1, v2, v3, v4 = corner_vectors
        coefficient = np.column_stack((v1, -v2, v3))
        a1, a2, a3 = np.linalg.solve(coefficient, v4)

        return np.array([a1, a2, a3, 1.0], dtype=float)

    def getUnscaledEdgeLengths(self, scaled_corner_vectors):
        edge_lengths = []

        for i in range(4):
            edge = scaled_corner_vectors[(i + 1) % 4] - scaled_corner_vectors[i]
            edge_lengths.append(np.linalg.norm(edge))

        return np.array(edge_lengths, dtype=float)

    def rotation_matrix_from_axis_angle(self, axis, angle_rad):
        axis = np.array(axis, dtype=float)
        axis = self.normalize(axis)

        x, y, z = axis

        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        t = 1 - c

        R = np.array([
            [t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c    ]
        ])

        return R

    def normalize(self, v):
        length = np.linalg.norm(v)

        if length == 0:
            return v

        return v / length
