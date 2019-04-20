import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
eyeDetect=cv2.CascadeClassifier('haarcascade_eye.xml');

cam=cv2.VideoCapture(0);

id=input('enter id')
sampleNum=0;
while True:
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    eyes = eyeDetect.detectMultiScale(gray,1.5,10);
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        cv2.imwrite("data/ User." +id+ "." +str(sampleNum)+ ".jpg" ,gray[y: y+h,x: x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
       # cv2.waitKey(100);
        cv2.imshow("Face",img);
    cv2.waitKey(1);
    if sampleNum>50:
        break
cam.release()
cv2.destroyAllWindows()
