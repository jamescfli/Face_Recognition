import photo_face_swap as pfs
import cv2
import numpy as np
import os

cam = cv2.VideoCapture(0)

face_dic = {'01': 'images/face_lib/01_WW_face.png',
            '02': 'images/face_lib/02_HH_face.png',
            '03': 'images/face_lib/03_DD_face.png',
            '04': 'images/face_lib/04_YH_face.png',
            '05': 'images/face_lib/05_LLY_face.png',    # very blurry
            '06': 'images/face_lib/06_LaLY_face.png',   # hair and eyebrow are mixed up
            '07': 'images/face_lib/07_ZY_face.png',
            '08': 'images/face_lib/08_YLJ_face.png',
            '09': 'images/face_lib/09_LK_face.png',
            '10': 'images/face_lib/10_LHY_face.png',
            '11': 'images/face_lib/11_ZYP_face.png',
            '12': 'images/face_lib/12_LCF_face.png',
            '13': 'images/face_lib/13_LCL_face.png',
            '14': 'images/face_lib/14_SHC_face.png',
            '15': 'images/face_lib/niuniu_face.jpg',
            '16': 'images/face_lib/bradpitt_face.jpg',
            '17': 'images/face_lib/bradpitt_body.jpg'}

# prepare face image
im2, landmarks2 = pfs.read_im_and_landmarks(face_dic['17'])  # face image

flag_save_video = True
flag_save_photo = False

if flag_save_video:
    # prepare save the video clips
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # overwrite
    filename = 'face_swap/output/output.avi'
    if os.path.isfile(filename):
        os.remove(filename)
    out = cv2.VideoWriter(filename,     # name of the output video file
                          fourcc,       # 4-character code of codec used to compress the frames
                          15.0,         # fps ~ 25 frames, need to be consistent with cv2.waitKey(*)
                          (1280, 720))  # frame size can be changed

counter = 0
while True:
    ret, img = cam.read()

    # head image for face swap
    im1 = cv2.resize(img, (img.shape[1] * pfs.SCALE_FACTOR,
                           img.shape[0] * pfs.SCALE_FACTOR))
    landmarks1 = pfs.get_landmarks(im1)

    M = pfs.transformation_from_points(landmarks1[pfs.ALIGN_POINTS],
                                       landmarks2[pfs.ALIGN_POINTS])

    mask = pfs.get_face_mask(im2, landmarks2)
    warped_mask = pfs.warp_im(mask, M, im1.shape)
    combined_mask = np.max([pfs.get_face_mask(im1, landmarks1), warped_mask], axis=0)

    warped_im2 = pfs.warp_im(im2, M, im1.shape)
    warped_corrected_im2 = pfs.correct_colours(im1, warped_im2, landmarks1)

    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

    # regularize frame, for cv2.imshow
    output_im = np.clip(output_im, 0, 255).astype('u1')     # u1 = uint8

    print output_im.shape
    print output_im.dtype   # float64 (be4 reg)
    print 'range [{}, {}]'.format(output_im.min(), output_im.max())     # sometime out of range of 255

    cv2.putText(output_im,
                'Face Swap',                    # title
                (5, output_im.shape[0] - 5),    # bottom left
                cv2.FONT_HERSHEY_PLAIN,         # HERSHEY has no Chinese support
                2.5,                            # text scale from base (corresponding to screen size)
                (30, 180, 30),                  # color - dark green
                4,                              # thickness of text
                cv2.LINE_AA)                    # line type: Anti-Aliased

    # display
    cv2.imshow('Video', output_im)
    counter += 1
    if flag_save_photo:
        cv2.imwrite('face_swap/output/output_{:03d}.jpg'.format(counter), output_im)
    if flag_save_video:
        out.write(output_im)

    k = cv2.waitKey(15) & 0xff  # 40ms ~25fps
    if k == ord('q'):
        break

cam.release()
if flag_save_video:
    out.release()
cv2.destroyAllWindows()