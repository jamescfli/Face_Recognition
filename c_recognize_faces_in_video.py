__author__ = 'bsl'

import cv2
import cv2.face
import os
import numpy as np
from video_camera import VideoCamera
from face_detector import FaceDetector
import frame_operations as fo

def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    # cope with .DS_Store by 'if'
    people = [person for person in os.listdir("people/") if os.path.isdir("people/" + person)]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/" + person):
            images.append(cv2.imread("people/" + person + '/' + image, 0))
            labels.append(i)
    return (images, np.array(labels), labels_dic)

def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2)
        cv2.rectangle(image, (x + w_rm, y), (x + w - w_rm, y + h),
                              (150, 150, 0), 8)


if __name__ == '__main__':
    cv2.startWindowThread()

    # Load Images, load labels and train models
    images, labels, labels_dic = collect_dataset()

    rec_eig = cv2.face.createEigenFaceRecognizer()
    rec_eig.train(images, labels)

    # needs at least two people
    rec_fisher = cv2.face.createFisherFaceRecognizer()
    rec_fisher.train(images, labels)

    rec_lbph = cv2.face.createLBPHFaceRecognizer()
    rec_lbph.train(images, labels)

    print "Models Trained Succesfully"


    detector = FaceDetector(xml_path='xml/haarcascade_frontalface_alt.xml')
    webcam = VideoCamera()
    cv2.namedWindow("PyData Tutorial", cv2.WINDOW_AUTOSIZE)
    while True:
        frame = webcam.get_frame()
        faces_coord = detector.detect(frame, False) # detect more than one face
        if len(faces_coord):
            faces = fo.normalize_faces(frame, faces_coord) # norm pipeline
            for i, face in enumerate(faces): # for each detected face
                # # MinDistancePredictCollector module is not available for unknown reason
                # collector = cv2.face.MinDistancePredictCollector()
                # rec_lbph.predict(face, collector)
                # conf = collector.getDist()
                # pred = collector.getLabel()
                pred, conf = rec_lbph.predict(face)
                threshold = 140
                print "Prediction: " + labels_dic[pred].capitalize() + "\nConfidence: " + str(round(conf))
                if conf < threshold: # apply threshold
                    cv2.putText(frame, labels_dic[pred].capitalize(),
                                (faces_coord[i][0], faces_coord[i][1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
                else:
                    cv2.putText(frame, "Unknown",
                                (faces_coord[i][0], faces_coord[i][1]),
                                cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
            draw_rectangle(frame, faces_coord) # rectangle around face
        cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)
        cv2.imshow("PyData Tutorial", frame) # live feed in external
        if cv2.waitKey(40) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
    del webcam