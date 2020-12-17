import dlib
import cv2
import numpy as np

import os
path = os.path.dirname(__file__)

# create list for landmarks
# ALL = list(range(0, 68))
# RIGHT_EYEBROW = list(range(17, 22))
# LEFT_EYEBROW = list(range(22, 27))
# RIGHT_EYE = list(range(36, 42))
# LEFT_EYE = list(range(42, 48))
# NOSE = list(range(27, 36))
# MOUTH_OUTLINE = list(range(48, 61))
# MOUTH_INNER = list(range(61, 68))
# JAWLINE = list(range(0, 17)) #턱선


# create face detector, predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path+'/shape_predictor_68_face_landmarks.dat')


# create VideoCapture object (input the video)
# 0 for web camera
vid_in = cv2.VideoCapture(0)
# "---" for the video file
#vid_in = cv2.VideoCapture("baby_vid.mp4")

# capture the image in an infinite loop
# -> make it looks like a video
while True:
    # Get frame from video
    # get success : ret = True / fail : ret= False
    ret, image_o = vid_in.read()

   # resize the video
   # 사이즈가 변하면 pixel사이의 값을 결정을 해야 하는데, 
   # 이때 사용하는 것을 보간법(Interpolation method)입니다. 
   # 많이 사용되는 보간법은 사이즈를 줄일 때는 cv2.INTER_AREA , 
   # 사이즈를 크게할 때는 cv2.INTER_CUBIC , cv2.INTER_LINEAR 을 사용
    image = cv2.resize(image_o, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces (up-sampling=1)
    face_detector = detector(img_gray, 1)
    # the number of face detected
    print("The number of faces detected : {}".format(len(face_detector)))

    # loop as the number of face
    # one loop belong to one face
    for face in face_detector:
        # face wrapped with rectangle
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()),
                      (0, 0, 255), 3)

        # make prediction and transform to numpy array
        landmarks = predictor(image, face)  # 얼굴에서 68개 점 찾기

        #create list to contain landmarks
        landmark_list = []

        # append (x, y) in landmark_list
        for p in landmarks.parts():
            landmark_list.append([p.x, p.y])
            cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), -1)


    cv2.imshow('result', image)

    # wait for keyboard input
    key = cv2.waitKey(1)

    # if esc,
    if key == 27:
        break

vid_in.release()