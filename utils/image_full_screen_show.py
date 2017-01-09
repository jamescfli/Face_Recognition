import sys

# from PyQt4 import QtGui
#
#
# def show_image(image_path='s_pycharm.jpg'):
#     app = QtGui.QApplication(sys.argv)
#     pixmap = QtGui.QPixmap(image_path)
#     screen = QtGui.QLabel()
#     screen.setPixmap(pixmap)
#     screen.showFullScreen()
#     sys.exit(app.exec_())

import cv2


# OpenCV display image in fullscreen on Mac without white border
# MBP resolution 1440x900 and peripheral monitor 1600x900
def show_full_screen_image():
    while True:
        print 'loading images...'
        img = cv2.imread('images/PyCharm_Fullscreen.png')
        print img.shape     # (1800, 2880, 3)   MBP with more squared screen than Lenovo X24A
        img = cv2.resize(img, (1440, 900), interpolation=cv2.INTER_CUBIC)  # width goes first and then height
        # img = cv2.resize(img, (1600, 900), interpolation=cv2.INTER_CUBIC)   # Len X24A
        print img.shape     # (900, 1600, 3)
        cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("test", img)
        key=cv2.waitKey(0)
        if key==27:
            break

if __name__ == '__main__':
    show_full_screen_image()

    # import numpy as np
    # img = np.zeros((900, 1600))
    # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.imshow("window",img)
    # cv2.waitKey(0)