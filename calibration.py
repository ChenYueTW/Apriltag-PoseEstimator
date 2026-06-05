import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 120)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
# cap.set(cv2.CAP_PROP_EXPOSURE, 30)
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
# cap.set(cv2.CAP_PROP_GAIN, 10)

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_250
)

parameters = cv2.aruco.DetectorParameters()

while True:
    ret, image = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        
    cv2.imshow("Aruco calibration", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()