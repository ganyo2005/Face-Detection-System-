import cv2 as cv 
import numpy as np
people=['Akufo Addo', 'John Mahamma', 'Obaama', 'Trump']

haar_casscade=cv.CascadeClassifier("Hand/haar_face.xml")

img=cv.imread(r'C:\Users\23354\Pictures\Feedback\Trump\433888.jpg')

face_reconginizer=cv.face.LBPHFaceRecognizer_create()
face_reconginizer.read('face_trained.yml')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

face_rect=haar_casscade.detectMultiScale(gray,1.1,5)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
for (x,y,w,h) in face_rect:
    face_roi=gray[y:y+h,x:x+h]

    label,confidence=face_reconginizer.predict(face_roi)
    print(f"This is {people[label]} with confidence of {confidence}")
    cv.putText(gray,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.1,(0,255,0),2)
    cv.rectangle(gray,(x,y),(x+w,y+h),1.0,thickness=2)

cv.imshow("Gray image ",gray)
cv.imshow("image",face_roi)

cv.waitKey(0)
