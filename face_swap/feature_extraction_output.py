import photo_face_swap as pfs
import cv2
import numpy as np

im2, landmarks2 = pfs.read_im_and_landmarks('images/face_lib/bradpitt_body.jpg')
print landmarks2.__class__  # <class 'numpy.matrixlib.defmatrix.matrix'>

# draw circle one by one around feature points
for point in np.array(landmarks2):  # matrix in tuple can not be recognized by 'cv2.circle'
    cv2.circle(im2, tuple(point), 2, (0, 255, 0), 1)

cv2.imwrite('face_swap/output/output.png', im2)