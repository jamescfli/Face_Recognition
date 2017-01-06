import cv2
import numpy as np
import os
import math

import frame_operations as fo
from video_camera import VideoCamera
from face_detector import FaceDetector
# from IPython.display import clear_output


if __name__ == '__main__':
    # # Detect Face in a Photo
    # VC = VideoCamera()
    #
    # frame = VC.get_frame(in_grayscale=False)
    # cv2.waitKey(1000)         # pause for cam to wake-up
    # frame = VC.get_frame(in_grayscale=False)
    # fo.plt_show(frame, title="Face Detection")
    #
    # face_detector = FaceDetector('xml/haarcascade_frontalface_alt.xml')
    # faces_coord = face_detector.detect(frame, biggest_only=False)
    # print "Type: " + str(type(faces_coord))
    # print faces_coord
    # print "Length: " + str(len(faces_coord))
    #
    # if len(faces_coord) >= 1:
    #     cv2.imwrite('images/screen_shot_origin.jpg', frame)

    # load image
    frame = cv2.imread('images/screen_shot_origin.jpg')

    face_detector = FaceDetector('xml/haarcascade_frontalface_alt.xml')
    faces_coord = face_detector.detect(frame, biggest_only=False)
    print "Type: " + str(type(faces_coord))
    print faces_coord
    print "Length: " + str(len(faces_coord))

    # fo.draw_face_rectangle(frame, faces_coord)
    # fo.plt_show(frame, title="Detected Faces in Rectangle")
    # if len(faces_coord) >= 1:
    #     cv2.imwrite('images/screen_shot_w_rectangled_faces.jpg', frame)

    # fo.draw_face_ellipse(frame, faces_coord)
    # fo.plt_show(frame, title="Detected Faces in Ellipse")
    # if len(faces_coord) >= 1:
    #     cv2.imwrite('images/screen_shot_w_ellipsed_faces.jpg', frame)

    # # Detect Face in a Live Video - Not working!
    # # cv2.startWindowThread()
    # webcam = VideoCamera()
    # face_detector = FaceDetector('xml/haarcascade_frontalface_alt.xml')
    # # Issue: Exception Name: NSInvalidArgumentException
    # # Description: -[NSApplication _setup:]: unrecognized selector sent to instance 0x7f98f1ab0c00
    # try:
    #     while True:
    #         frame = webcam.get_frame()
    #         faces_coord = face_detector.detect(frame)
    #         fo.draw_face_ellipse(frame, faces_coord)
    #         cv2.imshow('Face Detector', frame)
    #         fo.plt_show(frame)
    #         if cv2.waitKey(60) & 0xFF == ord('q'):
    #             break
    # except KeyboardInterrupt:
    #     print "Live Video Interrupted"
    # # webcam.__del__()
    # cv2.destroyAllWindows()

    # take photos for the time being
    if len(faces_coord):
        faces = fo.cut_face_rectangle(frame, faces_coord)
        # faces = fo.cut_face_ellipse(frame, faces_coord)
        faces = fo.normalize_intensity(faces)
        faces = fo.resize(faces)
        fo.plt_show(faces[0])

    # face_bw = cv2.cvtColor(faces[0], cv2.COLOR_BGR2GRAY)
    # face_bw_eq = cv2.equalizeHist(face_bw)
    # # fo.plt_show(np.hstack((face_bw, face_bw_eq)), "Before    After")
    #
    # from matplotlib import pyplot as plt
    # f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    # ax1.hist(face_bw.flatten(),100)
    # ax2.hist(face_bw_eq.flatten(),100, color = 'r')
    # ax1.set_xlim([0,255])
    # # ax1.set_ylim([0,1000])
    # ax2.set_xlim([0, 255])
    # # ax2.set_ylim([0, 700])
    # ax1.set_yticklabels([])
    # ax2.set_yticklabels([])
    # ax2.set_xlabel('Pixel Intensity Grayscale')
    # f.subplots_adjust(hspace=0)
    # plt.show()