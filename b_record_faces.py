__author__ = 'bsl'

import os
import cv2
from video_camera import VideoCamera
import frame_operations as fo

if __name__ == '__main__':
    folder = "people/" + raw_input('Person: ').lower() # input name
    VC = VideoCamera()
    VC.set_face_detector(xml_path='xml/haarcascade_frontalface_alt.xml')

    VC.show_frame_w_face_save(person_folder=folder)
