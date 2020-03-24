# laptop camera feed
# web ip (camera) feed

# Import
import cv2
import os
import glob
import sys
from PIL import Image
from keras.preprocessing.image import img_to_array
import scipy as sp

import numpy as np
import time
import requests
import pickle
from random import randint


def letters_to_ascii(list):
    out = []
    for i in range(len(list)):
        out.append(ord(list[i]))
    return out


def lap_cam(hand,
            labels,
            dsize,
            ntrials,
            trial_len,
            return_xyt,
            filename='dat',
            folder='data/',
            write_to_file=False
            ):
    X = []
    y = []
    t = []

    trial = 0  # start of last trial
    index = 0  # frame index
    # key = sp.nan
    key = -1
    limit = len(hand)
    last_trial = 0
    reaction_speed = .1
    ignore = True
    wait_reaction = False

    video_capture = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    font = cv2.FONT_HERSHEY_SIMPLEX
    # out = cv2.VideoWriter(folder+filename+'_vid.mp4', fourcc, 20.0, dsize)
    # ------------------------------------------------------------------------

    print('reaction_speed', reaction_speed)
    print('trial length', trial_len)

    _, frame = video_capture.read()
    start = time.time()
    while _:
        ts = np.round(time.time() - start, 3)  # total ts
        diff = np.round(ts - last_trial, 3)  # trial ts
        key = -1
        # key = sp.nan
        # if type trial, wait for reaction_time to be over, then put the trigger
        if wait_reaction:  #
            if diff > reaction_speed and diff_last < reaction_speed:  # only use once when you pass the border of reaction_speed
                key = labels[ind]
                ignore_once = False
                wait_reaction = False

        if diff > trial_len and ignore_once == True:
            if int(trial / 2) == ntrials:
                break
            if trial % 2 != 0:
                # type trial!
                wait_reaction = True
                ind = randint(0, limit - 1)
                print('')
                sys.stdout.write('[' + hand[ind] + ']')
            trial += 1
            last_trial = ts
        ignore_once = True
        ######################################################################
        img = cv2.resize(frame, dsize)  # preprocessing ~ atm only resizing, consider canny, etc
        ######################################################################

        # cv2.imshow('Cam', img)
        # out.write(img)
        X.append(img_to_array(img))
        t.append(ts)
        y.append(key)

        # k = cv2.waitKey(100) & 0xff
        # if k == 27 or int(trial/a) == ntrials:
        _, frame = video_capture.read()
        index += 1
        diff_last = np.round(t[index - 1] - last_trial)

    # ------------------------------------------------------------------------
    print('')
    print("Recording length:", np.round(time.time() - start, 2))
    # out.release()
    video_capture.release()
    cv2.destroyAllWindows()
    if return_xyt:
        return X, y, t
    else:
        return X, y


def web_cam(hand,
            labels,
            dsize,
            url,
            ntrials,
            trial_len,
            return_xyt,
            filename='dat',
            folder='data/'
            ):
    X = []
    y = []
    t = []

    # key = sp.nan
    key = -1
    trial = 0
    last_trial = 0  # start of last trial
    start = time.time()
    limit = len(hand)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    font = cv2.FONT_HERSHEY_SIMPLEX
    # out = cv2.VideoWriter(folder+filename+'_vid.mp4', fourcc, 20.0, dsize)
    # ------------------------------------------------------------------------

    while True:

        ts = np.round(time.time() - start, 3)  # total ts
        diff = np.round(ts - last_trial, 3)  # trial ts
        # key = sp.nan
        key = -1
        # if type trial, wait for reaction_time to be over, then put the trigger
        if wait_reaction:  #
            if diff > reaction_speed and diff_last < reaction_speed:  # only use once when you pass the border of reaction_speed
                key = labels[ind]
                ignore_once = False
                wait_reaction = False

        if diff > trial_len and ignore_once == True:
            if int(trial / 2) == ntrials:
                break
            if trial % 2 != 0:
                # type trial!
                wait_reaction = True
                ind = randint(0, limit - 1)
                print('')
                sys.stdout.write('[' + hand[ind] + ']')
            trial += 1
            last_trial = ts
        ignore_once = True

        ######################################################################
        imgreq = requests.get(url)
        imgarr = np.array(bytearray(imgreq.content), dtype=np.uint8)
        img = cv2.imdecode(imgarr, -1)
        img = cv2.resize(img, dsize)
        # imgt = cv2.putText(img, key, (10, 10), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        ######################################################################        

        # cv2.imshow('Cam', img)
        # out.write(img)
        X.append(img_to_array(img))
        y.append(key)
        t.append(ts)

        k = cv2.waitKey(100) & 0xff
        # if k == 27 or int(trial/a) == ntrials:
        #    break
        if int(trial / 2) == ntrials:
            break
    # ------------------------------------------------------------------------
    print('')
    print("Recording length:", np.round(time.time() - start, 2))
    # out.release()
    cv2.destroyAllWindows()

    if return_xyt:
        print('yolo')
        return X, y, t
    else:
        return X, y


def rec(ntrials=90,
        xdim=224,
        ydim=224,
        trial_len=.4,
        left=True,
        right=False,
        web=False,
        return_xyt=True,
        url='http://10.5.68.32:8080/shot.jpg',
        filename='rec',
        folder='data/',
        save=True
        ):
    out = []
    labels = []
    hand = []
    left_hand = ['a', 's', 'd', 'f', ' ']
    right_hand = ['j', 'k', 'l', 'z', ' ']

    if left:
        hand += left_hand
    if right:
        hand += right_hand
    labels = letters_to_ascii(hand)

    dsize = (xdim, ydim)

    if web:
        out = web_cam(hand=hand,
                      labels=labels,
                      dsize=dsize,
                      ntrials=ntrials,
                      url=url,
                      trial_len=trial_len,
                      return_xyt=return_xyt,
                      filename=filename,
                      folder=folder
                      )

    else:
        out = lap_cam(hand=hand,
                      labels=labels,
                      dsize=dsize,
                      ntrials=ntrials,
                      trial_len=trial_len,
                      return_xyt=return_xyt,
                      filename=filename,
                      folder=folder
                      )

    if save:
        date = str(time.time())
        with open(folder + filename + date, 'wb') as f:
            pickle.dump([out[0], out[1]], f)
    if return_xyt:
        return out[0], out[1], out[2]
    else:
        return out[0], out[1]
