""" This module contains the class VideoCamera, this classs provides us with
automtic functions to turn on the camera, record and turn off the camera
in the correct way.
"""

import cv2
from face_detector import FaceDetector
import frame_operations as fo
import os

class VideoCamera(object):
    """ A class to handle the video stream.
    """
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):  # called when the instance is about to be destroyed, e.g. program terminated
        self.video.release()

    def get_frame(self, in_grayscale=False):
        """ Get current frame of a live video.

        :param in_grayscale: Frame captured in color or grayscale [False].
        :type in_grayscale: Logical
        :return: Current video frame
        :rtype: numpy array
        """
        _, frame = self.video.read()

        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def show_frame(self, seconds, in_grayscale=False, size=(480, 360)):
        """ Show the frame of the live video.

        This function will show the current frame of the live video during
        the specified seconds. The frame is displayed in an external window.
        It also captures the key pressed during the time the frame was shown.
        This key can be used as an action indicator from the user.

        :param seconds: Amount of seconds the frame should be displayed.
        :param in_grayscale: Frame captured in color or grayscale [False].
        :type seconds: Double
        :type in_grayscale: Logical
        :return: Key pressed during the time the frame is shown
        :rtype: Integer
        """
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.shape < size:  # tuples are compared position by position
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        else:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('SnapShot', frame)
        key_pressed = cv2.waitKey(int(seconds * 1000))   # unit in milliseconds

        return key_pressed & 0xFF

    def set_face_detector(self, xml_path=None):
        if xml_path is not None:
            self.face_detector = FaceDetector(xml_path)
        else:
            print 'Face detector xml could not be identified ..'

    def show_frame_w_faces(self, seconds, box_style='bbox', size=(480, 360)):
        _, frame = self.video.read()
        # detect faces
        faces_coord = self.face_detector.detect(frame, biggest_only=False)
        # print "# of detected faces: {}".format(len(faces_coord))
        # draw rectangle or ellipse around the faces
        if box_style == 'bbox':
            fo.draw_face_rectangle(frame, faces_coord)
        elif box_style == 'ellipse':
            fo.draw_face_ellipse(frame, faces_coord)
        faces = fo.normalize_faces(frame, faces_coord)
        if len(faces) > 6:
            faces = faces[0:6]  # display the first 6 faces
        foo_face = cv2.imread('images/icon_user.png')
        foo_face = cv2.cvtColor(foo_face, cv2.COLOR_BGR2GRAY)
        foo_face = cv2.resize(foo_face, (120, 120), interpolation=cv2.INTER_CUBIC)

        face_board = faces + [foo_face]*(6-len(faces))
        face_board = cv2.hconcat((cv2.vconcat(face_board[0:3]),cv2.vconcat(face_board[3:6])))
        face_board = cv2.cvtColor(face_board, cv2.COLOR_GRAY2BGR)

        # resize view
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
        # concatenate with faces
        frame_final = cv2.hconcat((frame, face_board))

        cv2.imshow('SnapShots', frame_final)
        key_pressed = cv2.waitKey(int(seconds * 1000))   # unit in milliseconds

        return key_pressed & 0xFF

    def show_frame_w_face_save(self, person_folder=None):
        if not os.path.exists(person_folder):
            os.mkdir(person_folder)
            counter = 1
            timer = 0
            while counter < 21:
                _, frame = self.video.read()
                faces_coord = self.face_detector.detect(frame, biggest_only=True)
                if len(faces_coord) and timer % 700 == 50:  # every Second or so
                    faces = fo.normalize_faces(frame, faces_coord) # norm pipeline
                    cv2.imwrite(person_folder + '/{:03d}.jpg'.format(counter), faces[0])
                    print "Images Saved: {}".format(counter)
                    counter += 1
                fo.draw_face_rectangle(frame, faces_coord)
                cv2.imshow('Saving Faces', frame)
                cv2.waitKey(50)
                timer += 50
            cv2.destroyAllWindows()
        else:
            print "This name already exists."


if __name__ == '__main__':
    VC = VideoCamera()
    # frame = VC.get_frame()
    # print frame.shape   # (720, 1280, 3)
    VC.set_face_detector(xml_path='xml/haarcascade_frontalface_alt.xml')
    while True:
        KEY = VC.show_frame_w_faces(0.03, box_style='bbox')    # 0.03s delay ~= 33fps
        if KEY == 27:
            break
    VC.show_frame(1)    # better UE: freeze for one sec before going
