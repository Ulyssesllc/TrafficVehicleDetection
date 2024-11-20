from __future__ import print_function

import argparse
import glob
import os
import time

import matplotlib
import matplotlib.patches as patches

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import cdist
from skimage import io

np.random.seed(0)


# Hungarian algorithm
def linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def giou_batch(ltrbs_a, ltrbs_b, return_iou=False):
    """
    Compute the GIoU between two bboxes in the form [x1,y1,x2,y2]
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    ltrbs_b = np.expand_dims(ltrbs_b, 0)
    ltrbs_a = np.expand_dims(ltrbs_a, 1)

    xx1 = np.maximum(ltrbs_a[..., 0], ltrbs_b[..., 0])
    yy1 = np.maximum(ltrbs_a[..., 1], ltrbs_b[..., 1])
    xx2 = np.minimum(ltrbs_a[..., 2], ltrbs_b[..., 2])
    yy2 = np.minimum(ltrbs_a[..., 3], ltrbs_b[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    union = (
        (ltrbs_a[..., 2] - ltrbs_a[..., 0]) * (ltrbs_a[..., 3] - ltrbs_a[..., 1])
        + (ltrbs_b[..., 2] - ltrbs_b[..., 0]) * (ltrbs_b[..., 3] - ltrbs_b[..., 1])
        - wh
    )
    iou = wh / union

    xxc1 = np.minimum(ltrbs_a[..., 0], ltrbs_b[..., 0])
    yyc1 = np.minimum(ltrbs_a[..., 1], ltrbs_b[..., 1])
    xxc2 = np.maximum(ltrbs_a[..., 2], ltrbs_b[..., 2])
    yyc2 = np.maximum(ltrbs_a[..., 3], ltrbs_b[..., 3])
    wc = xxc2 - xxc1
    hc = yyc2 - yyc1
    assert (wc > 0).all() and (hc > 0).all()
    area_enclose = wc * hc
    giou = iou - (area_enclose - union) / area_enclose
    if return_iou:
        return giou, iou
    else:
        return giou


def cosine_distance_matrix(embeddings1, embeddings2):
    """
    Compute the cosine distance between the average feature embeddings and the incoming embeddings
    """
    distances = cdist(embeddings1, embeddings2, metric="cosine")
    return distances


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, feat):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 2.0
        self.kf.R[:2, :2] *= 1.0
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.1
        self.kf.Q[4:, 4:] *= 0.1

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.features = []
        self.features.append(feat)

    def update(self, bbox, feat):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.features.append(feat)
        if len(self.features) > 5:  # Limit to the last 10 features
            self.features.pop(0)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(
    detections, trackers, det_feats, trk_feats, giou_threshold=-0.5, reid_threshold=0.4, joint_threshold=1.1, alpha=0.3
):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """

    if len(trackers) == 0:

        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = giou_batch(detections, trackers)
    reid_matrix = cosine_distance_matrix(det_feats, trk_feats)
    combined_metrics = alpha * (1 - iou_matrix) + (1 - alpha) * reid_matrix

    if min(combined_metrics.shape) > 0:
        a = (combined_metrics < joint_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(combined_metrics)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []

    for m in matched_indices:
        if (combined_metrics[m[0], m[1]] > joint_threshold 
            or iou_matrix[m[0], m[1]] < giou_threshold
            or reid_matrix[m[0], m[1]] > reid_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=3, min_hits=0, alpha=0.3, giou_threshold=0.3, reid_threshold=0.45, joint_threshold=1):
        """
        Sets key parameters for SORT+ReID
        """
        self.max_age = max_age
        self.min_hits = min_hits

        self.giou_threshold = giou_threshold
        self.reid_threshold = reid_threshold
        self.alpha = alpha
        self.joint_threshold = joint_threshold

        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5)), feats=np.empty((0, 2048))):

        """
        Params:
            dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
            feats - a numpy array of features corresponding to the detections with shape (n, 2048)
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 5))
        trks_feat = np.zeros((len(self.trackers), 2048))

        to_del = []
        ret = []

        for t, trk in enumerate(trks):

            pos = self.trackers[t].predict()[0]
            trk[:5] = [pos[0], pos[1], pos[2], pos[3], 0]
            trks_feat[t, :] = np.array(self.trackers[t].features).mean(axis=0).reshape(1, -1)

            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, feats, trks_feat, self.giou_threshold, self.reid_threshold, self.joint_threshold, self.alpha
        )

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], feats[m[0], :])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], feats[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(
                    np.concatenate((d, [trk.id + 1])).reshape(1, -1)
                )  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
