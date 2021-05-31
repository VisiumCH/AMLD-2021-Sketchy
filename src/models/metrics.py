import math

import numpy as np
from joblib import Parallel, delayed

from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score


def get_similarity(sk_embeddings, im_embeddings):
    '''
    Sort images by similarity (closest to sketch) in the feature space
    The distance is computed as the euclidean distance (or cosine distance)
    It has reverse interpretation than the distance (closer is more similar and probable)
    Hence Similarity = 1/(1+distance)
    Args:
        - sk_embeddings: embeddings of the sketches [NxE]
        - im_embeddings: embeddings of the images [MxE]
    Return:
        - similarity: similarity value between images and sketches embeddings [NxM]
    '''
    return np.float32(1/(1 + cdist(np.float32(sk_embeddings),
                                   np.float32(im_embeddings), 'euclidean')))


def compare_classes(class_im, class_sk):
    '''
    Compare classes of images and sketches
    Args:
        - class_im: list of classes of the images [M]
        - class_sk: list of classes of the sketches [N]
    Return:
        - array [MxN] of 1 where the image and sketch belong to the same class and 0 elsewhere
    '''
    return (np.expand_dims(class_sk, axis=1) == np.expand_dims(class_im, axis=0)) * 1


def sort_by_similarity(similarity, class_matches):
    '''
    Sort the compared classes by decreasing similarity
    Args:
        - similarity: similarity value between images and sketches embeddings [NxM]
        - class_matches: 1 where the image and sketch belong to the same class and 0 elsewhere [NxM]
    return:
        - sorted_similarity: sorted sim [NxM]
        - sorted_class_matches: sorted class_matches [NxM]
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
    Precision at the place 200
    '''
    sorted_similarity, sorted_class_matches = sort_by_similarity(similarity, class_matches)

    nq = class_matches.shape[0]
    aps_200 = Parallel(n_jobs=num_cores)(delayed(average_precision_score)
                                         (sorted_class_matches[iq, 0:200], sorted_similarity[iq, 0:200])
                                         for iq in range(nq))
    aps_200_actual = [0.0 if math.isnan(x) else x for x in aps_200]
    map_200 = np.mean(aps_200_actual)

    # Precision@200 means at the place 200th
    prec_200 = np.mean(sorted_class_matches[:, 200])

    return map_200, prec_200


def get_map_all(similarity, class_matches, num_cores):
    '''
    Mean average precision considering all the data
    '''
    nq = class_matches.shape[0]
    ap_all = Parallel(n_jobs=num_cores)(delayed(average_precision_score)
                                        (class_matches[iq], similarity[iq]) for iq in range(nq))
    map_all = np.nanmean(ap_all)

    return ap_all, map_all
