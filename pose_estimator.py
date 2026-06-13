import numpy as np
import cv2
import math

CALIB_FILE = "camera.calib.npz"

data = np.load(CALIB_FILE)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

class PoseEstimator:
    def __init__(self):
        self.camera_pose = np.array([0.088957, 0.030321, 0.070595])
        self.camera_pitch = math.radians(-25.0) # degree
        self.central_sight = self.rotation_matrix_from_axis_angle(np.array([0, 1, 0]), self.camera_pitch) @ np.array([1.0, 0.0, 0.0])
        self.camera_x_axis = np.cross(self.central_sight, np.array([0.0, 0.0, 1.0]))
        self.camera_y_axis = np.cross(self.camera_x_axis, self.central_sight)
        self.apriltag_side_length = 0.1651
        
    def getTargetVector(self, tx, ty):
        target_x = self.rotation_matrix_from_axis_angle(self.camera_y_axis, math.radians(-tx)) @ self.central_sight
        target_y = self.rotation_matrix_from_axis_angle(self.camera_x_axis, math.radians(ty)) @ self.central_sight
        target = np.cross(np.cross(self.camera_y_axis, target_x), np.cross(self.camera_x_axis, target_y))

        return target
    
    def getTargetVectorFromPixel(self, x, y):
        point = np.array(
            [[[x, y]]],
            dtype=np.float32
        )
        
        normalized = cv2.undistortPoints(
            point,
            camera_matrix,
            dist_coeffs
        )

        x = normalized[0][0][0]
        y = normalized[0][0][1]

        tx = math.degrees(math.atan(x))
        ty = -math.degrees(math.atan(y))

        return self.getTargetVector(tx, ty)
    
    def getApriltagPose(self, target_vectors):
        # Get A
        v1 = target_vectors[0]
        v2 = target_vectors[1]
        v3 = target_vectors[2]
        v4 = target_vectors[3]

        B = np.array([[v1[0], -v2[0], v3[0]],
                      [v1[1], -v2[1], v3[1]],
                      [v1[2], -v2[2], v3[2]]])

        A = np.linalg.inv(B) @ v4
        A = np.append(A, 1)

        # Caculate lambda
        l1 = np.linalg.norm(A[1] * v2 - A[0] * v1)
        l2 = np.linalg.norm(A[2] * v3 - A[1] * v2)
        l3 = np.linalg.norm(A[3] * v4 - A[2] * v3)
        l4 = np.linalg.norm(A[0] * v1 - A[3] * v4)
        l_average = np.mean(np.array([l1, l2, l3, l4]))
        lambda_length = self.apriltag_side_length  / l_average

        t = lambda_length * A

        poses = np.zeros((4, 3))

        for i in range(4):
            poses[i] = self.camera_pose + t[i] * target_vectors[i]

        return poses

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
