#This Facedetaction example 
# Version_1
#-Created_by_Deepak_yadav
#In this example simply we detect face from image and drow rectangle around the fcace
#and print all axis(x,y,h,w) value.
#
import cv2

faceCascade= cv2.CascadeClassifier("data/cascades_data/haarcascade_frontalface_default.xml") #cascade File location

img=cv2.imread("data/space.jpg") #image file location

imgGray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting Gray scal image(Blaken-wite)
face=faceCascade.detectMultiScale(imgGray,1.1,4)
i=0
print(face)
for(x,y,w,h) in face:
    print(i,"=","x",x,"y",y,"w",w,"h",h)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #create Blue ractangle on around face 
    i=i+1
print("I Find ",i,"Face in this image")
cv2.imshow("Output",img) #show Face output
cv2.waitKey(0)  
cv2.destroyAllWindows()
