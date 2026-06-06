from pose_estimator import PoseEstimator

pose_estimator = PoseEstimator()
pose = pose_estimator.getPose(10, 10)
print(f"{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}")