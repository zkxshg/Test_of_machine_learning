# pip install opencv-python  
import cv2  
print(cv2.__version__) 

# ===================== read the image =====================
%matplotlib inline  
import numpy as np  
import cv2  
from matplotlib import pyplot as plt 

img = cv2.imread('/Ronaldo.jpg',1)  
img2 = cv2.imread('/Ronaldo.jpg',0)  
print(img2)  

# ===================== OpenCv to show the image =====================
cv2.imshow('image',img)  
cv2.waitKey(0)  
cv2.destroyAllWindows() 

img = cv2.imread('/Ronaldo.jpg',1)  
cv2.imwrite('Ronaldo.png',img)

# ===================== plt to show the image =====================
import numpy as np  
import cv2  
from matplotlib import pyplot as plt  
  
img = cv2.imread('/Ronaldo.jpg',0)  
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')  
plt.xticks([]), plt.yticks([])  
plt.show()  

# ===================== edit pixel =====================
import cv2  
import numpy as np  
img = cv2.imread('/Ronaldo.jpg')  
  
RGB=img[100,100]  
print(RGB)  
blue=img[100,100,0]  
print(blue)  

# ===================== edit region pixel =====================
img = cv2.imread('/Ronaldo.jpg')  
img2 = cv2.imread('/Ronaldo.jpg')  
img2[100:120,100:120] = [255,255,255]  
 
cv2.imshow('image_before',img)  
cv2.waitKey(0)  
cv2.destroyAllWindows()  
	  
cv2.imshow('image_after',img2)  
cv2.waitKey(0)  
cv2.destroyAllWindows() 

# ===================== Image Properties =====================
import cv2  
import numpy as np  
img = cv2.imread('/Ronaldo.jpg')  
print(img.shape)  
print(img.size)  
print(img.dtype)  

# ===================== Image ROI =====================
import cv2  
import numpy as np  
img = cv2.imread('/Ronaldo.jpg')  

ball = img[0:30,120:170]  
img[0:30, 200:250]=ball  
  
cv2.imshow('image_before',img)  
cv2.waitKey(0)  
cv2.destroyAllWindows()  

# ===================== arithmetic operations =====================
import cv2  
import numpy as np  
  
img1 = cv2.imread('/Ronaldo.jpg')  
img2 = img[:, 298::-1, :]  
	  
dst=cv2.addWeighted(img1,0.7,img2,0.3,0)  
cv2.imshow('dst',dst)  
cv2.waitKey(0)  
cv2.destroyAllWindow()  
