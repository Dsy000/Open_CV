import cv2
#read Webcam
cap=cv2.VideoCapture(0)  #0 means Laptop Default Web-cam
cap.set(3,640) #width-3
cap.set(4,480) #hight-4
cap.set(10,10) #britness-10

while True:
    success,img=cap.read()

    cv2.imshow("Video",img)
    #print(success)

    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
