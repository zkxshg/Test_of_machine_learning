# ================ play videos ================

# read the video
import numpy as np  
import cv2  
	  
# Initiate VideoCapture  
cap = cv2.VideoCapture('slow.mp4')  
print(cap.get(3),cap.get(4))  

# show the video
1while(cap.isOpened()):  
    ret, frame = cap.read()  
    
    if ret==True:        
       cv2.imshow('frame',frame)  
       if cv2.waitKey(50) & 0xFF == ord('q'):  
            break  
    else:  
        break  
         
cap.release()  
cv2.destroyAllWindows()  

# Video Writer
cap = cv2.VideoCapture('slow.mp4')  
  
# Define the codec and create VideoWriter object  
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,360))  
  
while(cap.isOpened()):  
    ret, frame = cap.read()  
    if ret==True:  
        # write the frame  
       out.write(frame)  
       cv2.imshow('frame',frame)  
       if cv2.waitKey(1) & 0xFF == ord('q'):  
           break  
   else:  
       break  
  
# Release everything if job is finished  
cap.release()  
out.release()  
cv2.destroyAllWindows()  

# ================ Background Subtraction ================
import numpy as np  
import cv2  

# read the video
cap = cv2.VideoCapture('vtest.avi')  
  
while(cap.isOpened()):  
    ret, frame = cap.read()  
      
    if ret==True:        
       cv2.imshow('frame',frame)  
       if cv2.waitKey(50) & 0xFF == ord('q'):  
           break  
   else:  
       break   
cap.release()  
cv2.destroyAllWindows() 

# Background Subtractor MOG2
import numpy as np
import cv2

cap = cv2.VideoCapture('vtest.avi')

fgbg = cv2.createBackgroundSubtractorMOG2()
while(1):
    ret, frame = cap.read()
    
    fgmask = fgbg.apply(frame)
    
    cv2.imshow('frame',fgmask)
    
    k = cv2.waitKey(30) & 0xff
    
    if k == 27:
        break
        
cap.release()
cv2.destroyAllWindows()

# ================ lucas kanade optical flow ================
import numpy as np
import cv2

cap = cv2.VideoCapture('slow.mp4')

# params for ShiTomasi corner detection
feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

cv2.imshow('old_gray',old_gray)    
cv2.waitKey(0)    
cv2.destroyAllWindows() 

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    
cv2.destroyAllWindows()
cap.release()

# ================ Meanshift and Camshift to track ================
import numpy as np
import cv2

cap = cv2.VideoCapture('slow.mp4')
# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 180,90,280,125 # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

cv2.imshow('roi',roi)    
cv2.waitKey(0)    
cv2.destroyAllWindows() 

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret,frame = cap.read()
    
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)
        
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)
    else:
        break
        
cv2.destroyAllWindows()
cap.release()

cap = cv2.VideoCapture('slow.mp4')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 180,90,280,125 # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow('img2',img2)
        
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)
    else:
        break

cv2.destroyAllWindows()
cap.release()

# stop  video
if cv2.waitKey(1) & 0xFF == ord('q'):  
    break  
    
k = cv2.waitKey(60) & 0xff  
if k == 27:  
    break  
