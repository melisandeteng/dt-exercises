#!/usr/bin/env python3

import numpy as np

from agent import PurePursuitPolicy
from utils import launch_env, seed
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask
import cv2
from PIL import Image

DATASET_DIR = "../dataset_duckietown_vf"

npz_index = 0


def save_npz(img, boxes, classes, labels):
    global npz_index
    with makedirs(DATASET_DIR):
        np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes, labels))
        print(f"{DATASET_DIR}/{npz_index}.npz", img)
        npz_index += 1


def remove_small_comps(im):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        im, None, None, None, 8, cv2.CV_32S
    )
    sizes = stats[1:, -1]  # get CC_STAT_AREA component
    img2 = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 50:  # filter small dotted regions
            img2[labels == i + 1] = 255
    return img2


def get_box_class(mask, c):
    boxes = []
    class_list = []
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )[-2:]
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, x + w, y + h])
        class_list.append(c)
    return boxes, class_list


def clean_segmented_image(seg_img, size=(224, 224)):
    # TODO
    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes
    # pass
    # return boxes, classes
    a = np.array(seg_img)
    sky = (np.all(a == (255, 0, 255), axis=-1) * 255).astype(np.uint8)
    duck = (np.all(a == (100, 117, 226), axis=-1) * 255).astype(np.uint8)
    cone = (np.all(a == (226, 111, 101), axis=-1) * 255).astype(np.uint8)
    truck = (np.all(a == (116, 114, 117), axis=-1) * 255).astype(np.uint8)
    bus = (np.all(a == (216, 171, 15), axis=-1) * 255).astype(np.uint8)

    boxes_list = []
    classes_list = []

    ducks = remove_small_comps(duck).astype(np.uint8)
    cones = remove_small_comps(cone).astype(np.uint8)
    trucks = remove_small_comps(truck).astype(np.uint8)
    buses = remove_small_comps(bus).astype(np.uint8)

    ducks = np.array(Image.fromarray(ducks).resize(size, resample=Image.NEAREST))
    cones = np.array(Image.fromarray(cones).resize(size, resample=Image.NEAREST))
    trucks = np.array(Image.fromarray(trucks).resize(size, resample=Image.NEAREST))
    buses = np.array(Image.fromarray(buses).resize(size, resample=Image.NEAREST))

    sky = np.array(Image.fromarray(sky).resize(size, resample=Image.NEAREST))
    b = []
    c = []
    boxes, class_list = get_box_class(ducks, 1)
    b += boxes
    c += class_list
    boxes, class_list = get_box_class(cones, 2)
    b += boxes
    c += class_list
    boxes, class_list = get_box_class(trucks, 3)
    b += boxes
    c += class_list
    boxes, class_list = get_box_class(buses, 4)
    b += boxes
    c += class_list

    im_array = np.array(Image.new("RGB", size))

    ducks_ = np.expand_dims(ducks, 2)
    ducks_ = np.repeat(ducks_, 3, axis=2) / 255
    ducks_ = np.multiply(ducks_, np.array([[100, 117, 226]])).astype(np.uint8)

    cones_ = np.expand_dims(cones, 2)
    cones_ = np.repeat(cones_, 3, axis=2) / 255
    cones_ = np.multiply(cones_, np.array([[226, 111, 101]])).astype(np.uint8)

    trucks_ = np.expand_dims(trucks, 2)
    trucks_ = np.repeat(trucks_, 3, axis=2) / 255
    trucks_ = np.multiply(trucks_, np.array([[116, 114, 117]])).astype(np.uint8)

    buses_ = np.expand_dims(buses, 2)
    buses_ = np.repeat(buses_, 3, axis=2) / 255
    buses_ = np.multiply(buses_, np.array([[216, 171, 15]])).astype(np.uint8)

    sky_ = np.expand_dims(sky, 2)
    sky_ = np.repeat(sky_, 3, axis=2) / 255
    sky_ = np.multiply(sky_, np.array([[255, 0, 255]])).astype(np.uint8)

    im_array = im_array + buses_ + trucks_ + ducks_ + cones_ + sky_

    labels = (
        ducks / 255 + cones / 255 * 2 + trucks / 255 * 3 + buses / 255 * 4
    ).astype(int)
    return (b, c, labels)


seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(
            action
        )  # Gives non-segmented obs as numpy array
        obs = np.array(Image.fromarray(obs).resize((224, 224)))
        segmented_obs = environment.render_obs(
            True
        )  # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)
        segmented_obs = Image.fromarray(segmented_obs)
        boxes, classes, labels = clean_segmented_image(segmented_obs)
        # TODO
        save_npz(obs, boxes, classes, labels)
        # save_npz(segmented_obs)
        print("saved obs")
        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break
