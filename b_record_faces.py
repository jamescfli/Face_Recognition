__author__ = 'bsl'

from utils.video_camera import VideoCamera

if __name__ == '__main__':
    folder = "people/" + raw_input('Person: ').lower() # input name
    VC = VideoCamera()
    VC.set_face_detector(xml_path='xml/haarcascade_frontalface_alt.xml')

    VC.show_frame_w_face_save(person_folder=folder)
