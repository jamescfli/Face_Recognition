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

def show_full_screen_image():
    var= 0
    while True:
        print 'loading images...'
        img = cv2.imread('images/PyCharm_Fullscreen.png')
        # print img.shape
        cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("test", img)
        key=cv2.waitKey(0)
        if key==27:
            break

if __name__ == '__main__':
    show_full_screen_image()