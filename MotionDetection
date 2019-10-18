import cv2
import numpy as numpy

cap = cv2.VideoCapture('test.avi')

#taking 1st two frames to find movement
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while(cap.isOpened()):
    #calculating difference between two frames to find movement
    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)

    #taking Gaussian Blur to eliminate noises caused by details and light
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    #dilating the threshold lines which comes abruptly
    dilated = cv2.dilate(thresh, None, iterations= 3)

    #finding contours or movements
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #function used to draw rectange
    for contour in contours:
        #getting the dimension of rectange
        (x,y,w,h) = cv2.boundingRect(contour)

        #eliminating contours which are unnecessary
        if cv2.contourArea(contour) < 900:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,"Status:{}".format('Movement'),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cv2.imshow("Frame", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
