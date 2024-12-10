import cv2;
import numpy as np
import os
# img=cv2.imread("shrreekrishna.jpg");
# imgs=cv2.resize(img,(400,500));
# cv2.imshow("Radhe radhw",img);

# v=np.array([1,2,3,4])
# h=np.hstack((v,v));
# print(h)
#

# h=np.vstack((imgs,imgs));
# v=np.hstack((h,h));
# cv2.imshow("Radhe radhw",v);
#
# cv2.waitKey(0);
listname=os.listdir(r"C:\Users\rosha\PycharmProjects\Computervision\images");
for name in listname:
    path=r"C:\Users\rosha\PycharmProjects\Computervision\images"
    img_name=path+"\\"+name
    img=cv2.imread(img_name)
    imgre=cv2.resize(img,(500,600))
    cv2.imshow('Radhe Shyam',imgre)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    print(img.shape)