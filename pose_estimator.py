import numpy as np
import cv2
import math

try:
    from scipy.optimize import least_squares as _least_squares
except Exception:  # pragma: no cover - scipy is a project dep, but degrade gracefully
    _least_squares = None

CALIB_FILE = "chessboard.calib.npz"

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

        # 相機座標基底的單位向量（前方 / 右 / 上），用於直接反投影成歸一化射線
        self.forward_hat = self.normalize(self.central_sight)
        self.x_hat = self.normalize(self.camera_x_axis)
        self.y_hat = self.normalize(self.camera_y_axis)

        self.apriltag_side_length = 0.161

    def getTargetVector(self, tx, ty):
        # 標準針孔反投影：方向 = 前方 + tan(方位角)·右 + tan(仰角)·上，再歸一化
        # tan(tx)=x_n、tan(ty)=y_n 即去畸變後的歸一化影像座標
        direction = (
            self.forward_hat
            + math.tan(math.radians(tx)) * self.x_hat
            + math.tan(math.radians(ty)) * self.y_hat
        )

        return self.normalize(direction)
    
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
    
    def getApriltagPose(self, target_vectors, center_vector=None):
        # 角點射線：P_i = camera_pose + (lambda * A_i) * v_i，A_i 為相對深度。
        v1 = target_vectors[0]
        v2 = target_vectors[1]
        v3 = target_vectors[2]
        v4 = target_vectors[3]

        if center_vector is None:
            # 原方法：僅用四角共面（平行四邊形）約束 P1 - P2 + P3 - P4 = 0，
            # 3x3 恰定解，對像素雜訊敏感（抖動大）。
            B = np.array([[v1[0], -v2[0], v3[0]],
                          [v1[1], -v2[1], v3[1]],
                          [v1[2], -v2[2], v3[2]]])

            A = np.linalg.inv(B) @ v4
            A = np.append(A, 1)
        else:
            # （選用）加入中心射線當作額外方程，以最小平方解超定系統。
            # 注意：只有當中心是「獨立且更精準」的量測時才會提升精度；若中心由
            # 四角推導（對角線交點 / 角點平均，多數偵測器如此），對平面四邊形而言
            # 中心 = 兩對角線交點，完全由四角決定，屬冗餘資訊，無法降低雜訊。
            # 未知數 x = [A0, A1, A2, dc']，A3 固定為 1 當尺度基準。
            #   平行四邊形：A0 v1 - A1 v2 + A2 v3            = v4
            #   中心點    ：A0 v1 + A1 v2 + A2 v3 - 4 dc' vc = -v4
            # （中心 C = (P1+P2+P3+P4)/4 同時落在中心射線 vc 上）
            vc = center_vector
            M = np.array([
                [v1[0], -v2[0], v3[0], 0.0],
                [v1[1], -v2[1], v3[1], 0.0],
                [v1[2], -v2[2], v3[2], 0.0],
                [v1[0],  v2[0], v3[0], -4.0 * vc[0]],
                [v1[1],  v2[1], v3[1], -4.0 * vc[1]],
                [v1[2],  v2[2], v3[2], -4.0 * vc[2]],
            ])
            b = np.array([v4[0], v4[1], v4[2], -v4[0], -v4[1], -v4[2]])
            sol, *_ = np.linalg.lstsq(M, b, rcond=None)
            A = np.array([sol[0], sol[1], sol[2], 1.0])

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

    def getApriltagPoseSquare(self, target_vectors):
        """Improved solve: fit the 4 corner-ray depths to a metric SQUARE.

        The original getApriltagPose enforces only coplanarity (a 3x3 exact
        inversion) and then scales by the mean side length. That linear system is
        ill-conditioned for a distant tag (condition number ~40 in practice), so
        sub-pixel corner noise is amplified ~10x into the depth/pose estimate —
        the cause of the large jitter and the metres-scale outliers.

        Here each corner is P_i = camera_pose + d_i * v_i (v_i the unit bearing,
        d_i the depth along the ray). We solve for the 4 depths that make the
        reconstructed quad the best-fit SQUARE of the known side length S: four
        equal sides + two equal diagonals (S*sqrt(2)). That is 6 residuals for 4
        unknowns -> over-determined and well-conditioned, so corner noise is
        averaged down instead of amplified. It uses the tag's known metric shape
        (square + size), which the coplanarity-only method never exploited, yet
        stays a 3D bearing-ray method, distinct from the IPPE image-plane homography.

        Returns the 4 corner positions in the world frame (same shape/order as
        getApriltagPose); falls back to getApriltagPose if scipy is unavailable.
        """
        v = np.asarray(target_vectors, dtype=float)

        # Initial depths from the cheap coplanarity solve (ball-park is enough;
        # the refinement below is robust to a poor start).
        B = np.array([[v[0][0], -v[1][0], v[2][0]],
                      [v[0][1], -v[1][1], v[2][1]],
                      [v[0][2], -v[1][2], v[2][2]]])
        try:
            A = np.linalg.solve(B, v[3])
            A = np.append(A, 1.0)
        except np.linalg.LinAlgError:
            A = np.ones(4)
        sides = [np.linalg.norm(A[1] * v[1] - A[0] * v[0]),
                 np.linalg.norm(A[2] * v[2] - A[1] * v[1]),
                 np.linalg.norm(A[3] * v[3] - A[2] * v[2]),
                 np.linalg.norm(A[0] * v[0] - A[3] * v[3])]
        l_avg = np.mean(sides)
        lam = self.apriltag_side_length / l_avg if l_avg > 0 else 1.0
        d0 = np.abs(lam * A)

        if _least_squares is None:
            return self.getApriltagPose(target_vectors)

        cam = self.camera_pose
        S = self.apriltag_side_length
        diag = S * math.sqrt(2.0)

        def residuals(d):
            P = cam + d[:, None] * v
            return [
                np.linalg.norm(P[0] - P[1]) - S,
                np.linalg.norm(P[1] - P[2]) - S,
                np.linalg.norm(P[2] - P[3]) - S,
                np.linalg.norm(P[3] - P[0]) - S,
                np.linalg.norm(P[0] - P[2]) - diag,
                np.linalg.norm(P[1] - P[3]) - diag,
            ]

        try:
            sol = _least_squares(residuals, d0, method="lm", max_nfev=50)
            d = sol.x
        except Exception:
            return self.getApriltagPose(target_vectors)

        return cam + d[:, None] * v

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
