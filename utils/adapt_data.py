import numpy as np
import random 
def batch_fill(testx, batch_size):
    """ Quick and dirty hack for filling smaller batch

    :param testx:
    :param batch_size:
    :return:
    """
    nr_batches_test = int(testx.shape[0] / batch_size)
    ran_from = nr_batches_test * batch_size
    ran_to = (nr_batches_test + 1) * batch_size
    size = testx[ran_from:ran_to].shape[0]
    new_shape = [batch_size - size]+list(testx.shape[1:])
    fill = np.ones(new_shape)
    return np.concatenate([testx[ran_from:ran_to], fill], axis=0), size

def adapt_labels_novelty_task(true_labels, label):
    """Adapt labels to anomaly detection context

    Args :
            true_labels (list): list of ints
            label (int): label which is considered anomalous
    Returns :
            true_labels (list): list of labels, 1 for anomalous and 0 for normal
    """
    if label == 0:
        (true_labels[true_labels == label], true_labels[true_labels != label]) = (0, 1)
        true_labels = [1] * true_labels.shape[0] - true_labels
    else:
        (true_labels[true_labels != label], true_labels[true_labels == label]) = (0, 1)

    return true_labels

def adapt_labels_outlier_task(true_labels, label):
    """Adapt labels to anomaly detection context

    Args :
            true_labels (list): list of ints
            label (int): label which is considered inlier
    Returns :
            true_labels (list): list of labels, 1 for anomalous and 0 for normal
    """
    #pdb.set_trace()
    ano_labels = np.zeros_like(true_labels) 
    idx_label = true_labels == label
    ano_labels[np.logical_not(idx_label)] = 1
    return ano_labels

def make_imbalance(majority_data, minority_data,ratio_minority, random_state=42):
    """Creates imbalanced data

    Args :
            majority_data (tuple of np.array): class of data which is supposed
                                               to be in majority
            minority_data (tuple of np.array): class of data which is supposed
                                               to be in minority
            ratio_minority (float) : proportion of minority data y
            random_state (int): (default 42) for random seed
    Returns :
            x (np.array): imbalanced data
            y (np.array): imbalanced labels
    """
    x_major, y_major = majority_data
    x_minor, y_minor = minority_data
    rng = np.random.RandomState(random_state)
    inds = rng.permutation(x_minor.shape[0])
    x_minor = x_minor[inds]
    y_minor = y_minor[inds]
    full_x_size = x_major.shape[0]
    sampling_minor_size = int(full_x_size * ratio_minority / (1 - ratio_minority))
    x_minor = x_minor[:sampling_minor_size]
    y_minor = y_minor[:sampling_minor_size]
    x = np.concatenate([x_major, x_minor], axis=0)
    y = np.concatenate([y_major, y_minor], axis=0)
    inds = rng.permutation(x.shape[0])

    return x[inds], y[inds]

def create_val_set(trainx, trainy, n_outliers):
    n_samples = int(n_outliers*(100/90))
    valx = []
    valy = []
    i = 0
    while len(valx) < n_outliers:
        if trainy[i] == 1:
            valx.append(trainx[i])
            valy.append(1)
        i += 1
    idx = set()
    del_idx = []
    while len(valx) < n_samples:
        j = random.randint(0, trainx.shape[0]-1)
        if (trainy[j] == 0) and (j not in idx) :
            valx.append(trainx[j])
            valy.append(0)
            idx.add(j)
            del_idx.append(j)
    c = list(zip(valx,valy))
    random.shuffle(c)
    valx, valy = zip(*c)
    trainx = np.delete(trainx, del_idx, axis=0)
    trainy = np.delete(trainy, del_idx, axis=0)
    return np.array(valx), np.array(valy), trainx, trainy



