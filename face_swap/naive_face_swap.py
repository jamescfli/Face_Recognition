#!/usr/bin/python
# coding:utf-8

import cv2
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

profile_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_alt.xml')

# detect face in prepared image
img = cv2.imread('images/1_yuzong_larger.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = profile_cascade.detectMultiScale(gray, 1.1, 5)
if (len(faces) >= 1):
    (x, y, w, h) = faces[0]
    roi_face_sub = img[y:y+h, x:x+w]
else:
    print 'no face found in preserved image ..'

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = profile_cascade.detectMultiScale(gray, 1.1, 5)

    if (len(faces) >= 1):
        # faces[0] will be adopted
        (x, y, w, h) = faces[0]
        roi_face_live = img[y:y+h, x:x+w]
        roi_face_patch = cv2.resize(roi_face_sub, (h, w), interpolation=cv2.INTER_AREA)
        img[y:y+h, x:x+w] = roi_face_patch

    cv2.putText(img,
                'Five-Mao Scum Special Effect',
                # '五毛渣特效'.encode('gbk'),
                (5, img.shape[0] - 5),  # bottom left
                cv2.FONT_HERSHEY_PLAIN, # HERSHEY has no Chinese support
                2.5,                    # text scale from base (corresponding to screen size)
                (30, 180, 30),          # color - dark green
                4,                      # thickness of text
                cv2.LINE_AA)            # line type
    cv2.imshow('Video', img)

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()