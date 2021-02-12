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


def preca_k(similarity, class_matches, k=None):
    '''
    Computes precision and recall (of k samples)
    '''
    act_lists = [np.nonzero(s)[0] for s in class_matches]
    nq = len(act_lists)
    pred_lists = np.argsort(-similarity, axis=1)

    num_cores = min(multiprocessing.cpu_count(), 8)
    prec_k = Parallel(n_jobs=num_cores)(delayed(precision)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    rec_k = Parallel(n_jobs=num_cores)(delayed(recall)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    return np.mean(prec_k), rec_k


def get_similarity(sk_embeddings, im_embeddings):
    '''
    Sort images by similarity (closest to sketch) in the feature space
    The distance is computed as the euclidean distance (or cosine distance)
    To later compute the precision, we need the probabilitity estimate of a class.
    It has reverse interpretation than the distance (closer is more similar and pobable)
    Hence Similarity = 1 - distance (or 1/(1+distance))
    Args:
        - sk_embeddings: embeddings of the sketches [NxE]
        - im_embeddings: embeddings of the images [NxE]
    Return:
        - similarity: similarity value between images and sketches embeddings [NxN]
    '''
    # Distance Measure
    distance = cdist(sk_embeddings, im_embeddings, 'euclidean')  # L1 same as Manhattan, Cityblock
    # distance = cdist(sk_embeddings, im_embeddings, 'cosine')/2 # Distance between 0 and 2

    # Similarity
    similarity = 1/(1+distance)
    # sim = 1 - distance

    return similarity


def compare_classes(class_im, class_sk):
    '''
    Compare classes of images and sketches
    Args:
        - class_im: list of classes of the images [N]
        - class_sk: list of classes of the sketches [N]
    Return:
        - array [NxN] of 1 where the image and sketch belong to the same class and 0 elsewhere
    '''
    return (np.expand_dims(class_sk, axis=1) == np.expand_dims(class_im, axis=0)) * 1


def sort_by_similarity(similarity, class_matches):
    '''
    Sort the compared classes by decreasing similarity
    Args:
        - similarity: similarity value between images and sketches embeddings [NxN]
        - class_matches: 1 where the image and sketch belong to the same class and 0 elsewhere [NxN]
    return:
        - sorted_similarity: sorted sim [NxN]
        - sorted_class_matches: sorted class_matches [NxN]
    '''
    arg_sorted_sim = (-similarity).argsort()
    sorted_similarity = []  # list of similarity values ordered by similarity (most to least similar)
    sorted_lst = []  # list of class target ordered by similarity (0 if different, 1 if same)
    for indx in range(0, arg_sorted_sim.shape[0]):
        sorted_similarity.append(similarity[indx, arg_sorted_sim[indx, :]])
        sorted_lst.append(class_matches[indx, arg_sorted_sim[indx, :]])

    sorted_similarity = np.array(sorted_similarity)
    sorted_class_matches = np.array(sorted_lst)

    return sorted_similarity, sorted_class_matches


def get_map_prec_200(similarity, class_matches, num_cores):
    '''
    Mean average precision considering the 200 most similar embeddings
    Precision at the pllace 200
    '''
    sorted_similarity, sorted_class_matches = sort_by_similarity(similarity, class_matches)

    nq = class_matches.shape[0]
    aps_200 = Parallel(n_jobs=num_cores)(delayed(average_precision_score)
                                         (sorted_class_matches[iq, 0:200], sorted_similarity[iq, 0:200])
                                         for iq in range(nq))
    aps_200_actual = [0.0 if math.isnan(x) else x for x in aps_200]
    map_200 = np.mean(aps_200_actual)

    # Precision@200 means at the place 200th
    prec_200 = np.mean(sorted_class_matches[:, 100])

    return map_200, prec_200


def get_map_all(similarity, class_matches, num_cores):
    '''
    Mean average precision considering all the data
    '''
    nq = class_matches.shape[0]
    ap_all = Parallel(n_jobs=num_cores)(delayed(average_precision_score)
                                        (class_matches[iq], similarity[iq]) for iq in range(nq))
    map_all = np.mean(ap_all)

    return ap_all, map_all
