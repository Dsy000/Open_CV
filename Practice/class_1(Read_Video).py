import cv2
#read Videos

cap=cv2.VideoCapture("Data/test_video.mp4")

while True:
    success,img=cap.read()

    cv2.imshow("Video",img)
    #print(success)
    
if cv2.waitKey(1) & 0xff ==ord('q'):
        break

