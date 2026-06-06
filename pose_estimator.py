import numpy as np
import math

class PoseEstimator:
    def __init__(self):
        self.camera_pose = np.array([0.088957, 0.030321, 0.070595])
        self.camera_pitch = math.radians(-25.0) # degree
        self.central_sight = self.rotation_matrix_from_axis_angle(np.array([0, 1, 0]), self.camera_pitch) @ np.array([1.0, 0.0, 0.0])
        self.camera_x_axis = np.cross(self.central_sight, np.array([0.0, 0.0, 1.0]))
        self.camera_y_axis = np.cross(self.camera_x_axis, self.central_sight)
        self.apriltag_height = 0.2
        
    def getPose(self, tx, ty):
        target_x = self.rotation_matrix_from_axis_angle(self.camera_y_axis, math.radians(-tx)) @ self.central_sight
        target_y = self.rotation_matrix_from_axis_angle(self.camera_x_axis, math.radians(ty)) @ self.central_sight
        target = np.cross(np.cross(self.camera_x_axis, target_y), np.cross(self.camera_y_axis, target_x))

        t = (self.apriltag_height - self.camera_pose[2]) / target[2]
        apriltag_pose = target * t + self.camera_pose

        return apriltag_pose

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