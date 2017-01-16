import photo_face_swap as pfs
import cv2
import numpy as np

# im2, landmarks2 = pfs.read_im_and_landmarks('images/face_lib/06_LaLY_face.png')
# im2, landmarks2 = pfs.read_im_and_landmarks('images/theta_face_lib/face_at_top_5376x2688.jpg')   # NoFaces
# im2, landmarks2 = pfs.read_im_and_landmarks('images/theta_face_lib/face_in_middle_5376x2688.jpg')   # with drift
im2, landmarks2 = pfs.read_im_and_landmarks('images/theta_face_lib/face_at_bottom_5376x2688.jpg')   # NoFaces
# print landmarks2.__class__  # <class 'numpy.matrixlib.defmatrix.matrix'> ==> np.matrix

# draw circle one by one around feature points
for point in np.array(landmarks2):  # matrix in tuple can not be recognized by 'cv2.circle'
    cv2.circle(im2, tuple(point), 2, (0, 255, 0), 1)

# save
cv2.imwrite('face_swap/output/output.png', im2)
# display
cv2.startWindowThread()
while True:
    cv2.imshow('Face Feature Extraction', im2)
    k = cv2.waitKey(1000) & 0xff
    if k == ord('q'):
        break
cv2.destroyAllWindows()