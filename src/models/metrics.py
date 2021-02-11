import math

import numpy as np
import multiprocessing
from joblib import Parallel, delayed

from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score


def recall(actual, predicted, k):
    '''
    Computes recall
    '''
    act_set = set(actual)
    pred_set = set(predicted[:k])
    re = len(act_set & pred_set) / len(act_set)
    return re


def precision(actual, predicted, k):
    '''
    Computes precision
    '''
    act_set = set(actual)
    if k is not None:
        pred_set = set(predicted[:k])
    else:
        pred_set = set(predicted)
    pr = len(act_set & pred_set) / min(len(act_set), len(pred_set))
    return pr


def precak(sim, str_sim, k=None):
    '''
    Camputes precision and recall (of k samples)
    '''
    act_lists = [np.nonzero(s)[0] for s in str_sim]
    nq = len(act_lists)
    pred_lists = np.argsort(-sim, axis=1)

    num_cores = min(multiprocessing.cpu_count(), 8)
    preck = Parallel(n_jobs=num_cores)(delayed(precision)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    reck = Parallel(n_jobs=num_cores)(delayed(recall)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    return np.mean(preck), reck


def get_similarity(acc_sk_em, acc_im_em):
    '''
    Sort images by similarity (closest to sketch) in the feature space
    The distance is computed as the euclidean distance (or cosine distance)
    To later compute the precision, we need the probabilitity estimate of a class.
    It has reverse interpretation than the distance (closer is more similar and pobable)
    Hence Similarity = 1 - distance (or 1/(1+distance))
    Args:
        - acc_sk_em: embeddings of the sketches [NxE]
        - acc_im_em: embeddings of the images [NxE]
    Return:
        - sim: similarity value between images and sketches embeddings [NxN]
    '''
    # Distance Measure
    distance = cdist(acc_sk_em, acc_im_em, 'euclidean')  # L1 same as Manhattan, Cityblock
    # distance = cdist(acc_sk_em, acc_im_em, 'cosine')/2 # Distance between 0 and 2

    # Similarity
    sim = 1/(1+distance)
    # sim = 1 - distance

    return sim


def compare_classes(acc_cls_im, acc_cls_sk):
    '''
    Compare classes of images and sketches
    Args:
        - acc_cls_im: list of classes of the images [N] 
        - acc_cls_sk: list of classes of the sketches [N]
    Return:
        - array [NxN] of 1 where the image and sketch belong to the same class and 0 elsewhere
    '''
    return (np.expand_dims(acc_cls_sk, axis=1) == np.expand_dims(acc_cls_im, axis=0)) * 1


def sort_by_similarity(sim, str_sim):
    '''
    Sort the compared classes by decreasing similarity
    Args:
        - sim: similarity value between images and sketches embeddings [NxN]
        - str_sim: 1 where the image and sketch belong to the same class and 0 elsewhere [NxN]
    return:
        - sort_sim: sorted sim [NxN]
        - sort_str_sim: sorted str_sim [NxN]
    '''
    arg_sort_sim = (-sim).argsort()
    sort_sim = []  # list of similarity values ordered by similarity (most to least similar)
    sort_lst = []  # list of class target ordered by similarity (0 if different, 1 if same)
    for indx in range(0, arg_sort_sim.shape[0]):
        sort_sim.append(sim[indx, arg_sort_sim[indx, :]])
        sort_lst.append(str_sim[indx, arg_sort_sim[indx, :]])

    sort_sim = np.array(sort_sim)
    sort_str_sim = np.array(sort_lst)

    return sort_sim, sort_str_sim


def get_map_prec_200(sim, str_sim, num_cores):
    '''
    Mean average precision considering the 200 most similar embeddings
    Precision at the pllace 200
    '''
    sort_sim, sort_str_sim = sort_by_similarity(sim, str_sim)

    nq = str_sim.shape[0]
    aps_200 = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(sort_str_sim[iq, 0:200], sort_sim[iq, 0:200])
                                         for iq in range(nq))
    aps_200_actual = [0.0 if math.isnan(x) else x for x in aps_200]
    map_200 = np.mean(aps_200_actual)

    # Precision@200 means at the place 200th
    prec_200 = np.mean(sort_str_sim[:, 100])

    return map_200, prec_200


def get_map_all(sim, str_sim, num_cores):
    '''
    Mean average precision considering all the data
    '''
    nq = str_sim.shape[0]
    ap_all = Parallel(n_jobs=num_cores)(delayed(average_precision_score)(str_sim[iq], sim[iq]) for iq in range(nq))
    map_all = np.mean(ap_all)

    return ap_all, map_all
