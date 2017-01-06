__author__ = 'bsl'

from utils.video_camera import VideoCamera


if __name__ == '__main__':
    VC = VideoCamera()
    # frame = VC.get_frame()
    # print frame.shape   # (720, 1280, 3)
    VC.set_face_detector(xml_path='xml/haarcascade_frontalface_alt.xml')
    while True:
        KEY = VC.show_frame_w_faces(0.03, box_style='bbox')    # 0.03s delay ~= 33fps
        if KEY == 27:
            break
    VC.show_frame_w_faces(1)
