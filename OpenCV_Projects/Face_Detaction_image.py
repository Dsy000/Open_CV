#This Facedetaction example 
#
import cv2

faceCascade= cv2.CascadeClassifier("data/cascades_data/haarcascade_frontalface_default.xml") #cascade File location

img=cv2.imread("data/space.jpg") #image file location

imgGray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting Gray scal image(Blaken-wite)
face=faceCascade.detectMultiScale(imgGray,1.1,4)

print(face)
for(x,y,w,h) in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #create Blue ractangle on around face 

cv2.imshow("Output",img) #show Face output
cv2.waitKey(0)  
