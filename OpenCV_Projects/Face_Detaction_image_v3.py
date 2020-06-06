# Version_3
#in this version we cropt face and show on new window.
#-Created_by_Deepak_yadav
#
import cv2

faceCascade= cv2.CascadeClassifier("data/cascades_data/haarcascade_frontalface_default.xml") #cascade File location

img=cv2.imread("data/space.jpg") #image file location

img2=img.copy()
imgGray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting Gray scal image(Blaken-wite)
face=faceCascade.detectMultiScale(imgGray,1.1,4)

#print(face)
i=0
for(x,y,w,h) in face:
    print("This Is image ",i," area_coordination")
    print(i,"-->","x=",x,"y=",y,"w=",w,"h=",h)

    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.putText(img,"{}".format(i) ,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(0, 0, 255),1)
    imgCropped = img2[y:y+h, x:x+h]  # [Hight,width]
    cv2.imshow("Image {}".format(i), imgCropped) #Show all face deferant windows
    i=i+1

cv2.imshow("Output",img)
print("I am Find ",i,"Face in this image")
cv2.waitKey(0)
cv2.destroyAllWindows()
