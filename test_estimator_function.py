from pose_estimator import PoseEstimator

pose_estimator = PoseEstimator(0.16)
pose = pose_estimator.getCornerPoses(
    10,
    10,
    [
        (8, 8),
        (12, 8),
        (12, 12),
        (8, 12),
    ]
)

print(f"center: {pose['center_pose'][0]:.2f}, {pose['center_pose'][1]:.2f}, {pose['center_pose'][2]:.2f}")
for i, corner in enumerate(pose["corner_poses"], start=1):
    print(f"P{i}: {corner[0]:.2f}, {corner[1]:.2f}, {corner[2]:.2f}")
