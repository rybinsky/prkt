import cv2

cap = cv2.VideoCapture("http://185.137.146.14/mjpg/video.mjpg")
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
