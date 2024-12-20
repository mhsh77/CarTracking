from ultralytics import YOLO
import cv2 #opencv
import math
import cvzone #this lib is used for some common perpuses in computer vision projects
from sort import *

model = YOLO('yolov8n.pt') #getting the yolo8nano which is fast and efficent for real time applications
cap = cv2.VideoCapture('CarCounter\Videos\cars.mp4') # getting access to the camera number 0
mask = cv2.imread("CarCounter\mask.png")
template = cv2.imread("CarCounter\cover.png",cv2.IMREAD_UNCHANGED)
template = cv2.resize(template, (200, 100))
mask = cv2.resize(mask, (700, 700)) 
totalCount = []
tracker = Sort(max_age=20, min_hits=3,iou_threshold=0.3)
limits = [200, 0, 200, 699]
while True:
    _,img = cap.read() # getting the data from the camera
    # Resize the image frames 
    img = cv2.resize(img, (700, 700)) 
    imgMasked=cv2.bitwise_and(img,mask)
    results = model(imgMasked,stream=True) # use the model on images and get results
    detections = np.empty((0,5))
    for r in results: # draw boxes around detected objects write their flag 
        boxes = r.boxes
        for box in boxes:
            cls = box.cls[0]
            conf = math.ceil((box.conf[0] * 100)) / 100
            if model.names[int(cls.item())]=='car' or model.names[int(cls.item())] == 'truck' or model.names[int(cls.item())] == 'bus' or model.names[int(cls.item())] == 'motorbike'\
            and  conf > 0.5:
                print(cls.item()) # get the predicted class
                
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                w, h = x2 - x1, y2 - y1
                if conf > 0.6:
                    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5,colorR=(0,0,0))
                    cvzone.putTextRect(img, f'car {conf}', (max(0, x1), max(35, y1)),
                                    scale=1, thickness=1, offset=0,colorR=(0,0,0))
                    
                    currentArray = np.array([x1,y1,x2,y2,math.ceil(box.conf[0]*100)/100])
                    detections = np.vstack((detections,currentArray))
    resultsTracker = tracker.update(detections)  
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)  
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
 
        if limits[0] - 15 < cx < limits[0] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    cvzone.overlayPNG(img,template)

    cv2.putText(img,f'{len(totalCount)}',(140,44),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
 
    cv2.imshow("image",img) # show image
    cv2.waitKey(1) # wait one milsec to show img in realtime