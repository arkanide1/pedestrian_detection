import cv2 as cv

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
video = "Resources\Videos\dog.mp4"
cap = cv.VideoCapture(video)


while cap.isOpened() :
    _ , frame = cap.read()
    (rects, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv.putText(frame ,'pedestrian' , (x , y-10), 20, 0.5, (0, 255, 0 ))
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow('Video', frame)

    # Wait for a key press, and exit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()