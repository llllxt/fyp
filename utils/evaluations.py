from __future__ import absolute_import

import os
import csv
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn import manifold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_recall_fscore_support
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import time
import matplotlib.cm as cm
from constants import IMAGES_DATASETS
from prg import prg
import itertools
from itertools import cycle
from scipy import interp
# import cv2
sns.set(color_codes=True)


def plot_confusion_matrix(cm, classes=[],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, direc="results",step=-1):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.Figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if classes:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(direc, str(step)))
    plt.close()


def do_roc(scores, true_labels, file_name='', directory='', plot=False):
    """ Does the ROC curve

    Args:
            scores (list): list of scores from the decision function
            true_labels (list): list of labels associated to the scores
            file_name (str): name of the ROC curve
            directory (str): directory to save the jpg file
            plot (bool): plots the ROC curve or not
    Returns:
            roc_auc (float): area under the under the ROC curve
            thresholds (list): list of thresholds for the ROC
    """
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr) # compute area under the curve
    if plot: 
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        plt.savefig(directory + file_name + 'roc.png')
        plt.close()

    return roc_auc

def do_prc(scores, true_labels, file_name='', directory='', plot=False):
    """ Does the PRC curve

    Args :
            scores (list): list of scores from the decision function
            true_labels (list): list of labels associated to the scores
            file_name (str): name of the PRC curve
            directory (str): directory to save the jpg file
            plot (bool): plots the ROC curve or not
    Returns:
            prc_auc (float): area under the under the PRC curve
            pre_score (float): precision score
            rec_score (float): recall score
            F1_score (float): max F1 score according to different thresholds
    """
    precision, recall, _ = precision_recall_curve(true_labels, scores)

    prc_auc = auc(recall, precision)

    if plot:
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: AUC=%0.4f'
                            %(prc_auc))
        plt.savefig(directory + file_name + 'prc.png')
        plt.close()

    return prc_auc

def do_prg(scores, true_labels, file_name='', directory='', plot=False):
    prg_curve = prg.create_prg_curve(true_labels, scores)
    auprg = prg.calc_auprg(prg_curve)
    if plot:
       prg.plot_prg(prg_curve)
       plt.title('Precision-Recall-Gain curve: AUC=%0.4f'
                            %(auprg))
       plt.savefig(directory + file_name + "prg.png")


    return auprg

def do_cumdist(scores, file_name='', directory='', plot=True):
    N = len(scores)
    X2 = np.sort(scores)
    F2 = np.array(range(N))/float(N)
    if plot:
        plt.figure()
        plt.xlabel("Anomaly score")
        plt.ylabel("Percentage")
        plt.title("Cumulative distribution function of the anomaly score")
        plt.plot(X2, F2)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + file_name + 'cum_dist.png')

def get_percentile(scores, dataset, anomaly_type, anomaly_proportion):
        # Threshold was set to 5% for the recall results in the supplements
        #c = anomaly_proportion * 100
        # Highest 5% are anomalous
        per = np.percentile(scores, 100 - 5)
        return per



def save_insights(scores, true_labels, latent, rec_error, reconstructions, model, dataset, method, weight, label,
                 random_seed, anomaly_type, anomaly_proportion, step=-1, header=0):


    directory = 'insights/{}/{}/{}_{}_{}_w{}/l{}'.format(model,
                                                  dataset,
                                                  anomaly_type,
                                                  anomaly_proportion,
                                                  method,
                                                  weight, 
                                                  label)

    if not os.path.exists(directory):
        os.makedirs(directory)

    if dataset in IMAGES_DATASETS:
        np.save(directory+"reconstructions_{}.npy".format(random_seed), reconstructions)
        np.save(directory+"latent_{}.npy".format(random_seed), latent)


    scores = np.array(scores) 
    n = scores.shape[0]
    if latent is not None:
        latent_norm = np.linalg.norm(latent, axis=1)

        sum_lat = np.sum(latent_norm)  
        print("Sum of latent norm is ", sum_lat)
    
    per = get_percentile(scores, dataset, anomaly_type, anomaly_proportion)    
    y_pred = scores>=per
    for i in range(true_labels.shape[0]):
        if dataset in IMAGES_DATASETS and header==0:
            results = [model, dataset, anomaly_type, anomaly_proportion, method, weight, label,
                      step, i, latent_norm[i], scores[i], true_labels[i], y_pred[i]]
            save_results_csv("insights/scores.csv", results, header=4)
        else:
            results = [model, dataset, method, weight, label,
                      true_labels[i], scores[i]]
            save_results_csv("insights/scores.csv", results, header=5)


def do_hist(scores, true_labels, directory, dataset, random_seed, display=False):
    plt.figure()
    idx_inliers = (true_labels == 0)
    idx_outliers = (true_labels == 1)
    hrange = (min(scores), max(scores))
    plt.hist(scores[idx_inliers], 50, facecolor=(0, 1, 0, 0.5),
             label="Normal samples", density=False, range=hrange)
    plt.hist(scores[idx_outliers], 50, facecolor=(1, 0, 0, 0.5),
             label="Anomalous samples", density=False, range=hrange)
    plt.title("Distribution of the anomaly score")
    plt.legend()
    if display:
       plt.show()
    else:
        plt.savefig(directory + 'histogram_{}_{}.png'.format(random_seed, dataset),
                    transparent=True, bbox_inches='tight')
        plt.close()


def do_hists(scores, true_labels, directory, dataset, random_seed, display=False):
    plt.figure()
    n_samples = len(scores)
    n_labels = np.max(true_labels)
    hrange = (min(scores), max(scores))
    for l in range(n_labels+1):
       idx = (true_labels == l)
       plt.hist(scores[idx], 50, facecolor=(int(l==0), int(l==1), int(l==2), 1),
                label="{}".format(l), density=False, range=hrange)
    plt.title("Distribution of the anomaly score")
    plt.legend()
    if display:
       plt.show()
    else:
        plt.savefig(directory + 'hists_{}_{}.png'.format(random_seed, dataset),
                    transparent=True, bbox_inches='tight')
        plt.close()



def predict(scores, threshold):
    return scores>=threshold

def make_meshgrid(x_min,x_max,y_min,y_max, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def save_grid_plot(samples, samples_rec, name_model, dataset, nb_images,
                   grid_width=10):

    args = name_model.split('/')[:-1]
    directory = os.path.join(*args)
    if not os.path.exists(directory):
        os.makedirs(directory)
    samples = (samples + 1) / 2
    samples_rec = (samples_rec + 1) / 2
    if dataset == 'mnist':
        figsize = (28,28)
    elif dataset == 'rop':
        figsize = (128,128)
    else:
        figsize = (32, 32)
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(grid_width, grid_width)
    gs.update(wspace=0.05, hspace=0.05)
    list_samples = []
    ax = plt.subplot()
    plt.imshow(sample)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.savefig('{}.png'.format(name_model))        
    for x, x_rec in zip(np.split(samples, nb_images // grid_width),
                        np.split(samples_rec, nb_images // grid_width)):
        list_samples += np.split(x, grid_width) + np.split(x_rec, grid_width)
    list_samples = [np.squeeze(sample) for sample in list_samples]
    for i, sample in enumerate(list_samples):
        if i>=nb_images*2:
            break
        ax = plt.subplot(gs[i])
        if dataset == 'mnist':
            plt.imshow(sample, cmap=cm.gray)
        else:
            plt.imshow(sample)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    plt.savefig('{}.png'.format(name_model))




def save_results(scores, true_labels, model, dataset, method, weight, label,
                 random_seed, anomaly_type, anomaly_proportion, step=-1):

    directory = 'results/{}/{}/{}_{}/{}/w{}/'.format(model,
                                                  dataset,
                                                  anomaly_type,
                                                  anomaly_proportion,
                                                  method,
                                                  weight)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if dataset in IMAGES_DATASETS:
        file_name = "{}_step{}_rd{}".format(label, step, random_seed)
        fname = directory + "{}.csv".format(label)
    else:
        file_name = "{}_step{}_rd{}".format(dataset, step, random_seed)
        fname = directory + "results.csv"

    
    scores = np.array(scores) 
   
    roc_auc = do_roc(scores, true_labels, file_name=file_name,
                    directory=directory)
    prc_auc = do_prc(scores, true_labels, file_name=file_name,
                        directory=directory)
    do_cumdist(scores, file_name=file_name, directory=directory)
    prg_auc = do_prg(scores, true_labels, file_name=file_name,
                        directory=directory)
    do_hist(scores, true_labels, directory, dataset, random_seed)

    per = get_percentile(scores, dataset, anomaly_type, anomaly_proportion)    
    y_pred = (scores>=per)
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels.astype(int),
                                                               y_pred.astype(int),
                                                               average='binary')

    print("Testing at step %i, method %s: Prec = %.4f | Rec = %.4f | F1 = %.4f"
        % (step, method, precision, recall, f1))

    print("Testing method {} | ROC AUC = {:.4f} | PRC AUC = {:.4f} | PRG AUC = {:.4f}".format(method, roc_auc,
                                                                                              prc_auc, prg_auc))

    results = [model, dataset, anomaly_type, anomaly_proportion, method, weight, label,
               step, roc_auc, prc_auc, prg_auc, precision, recall, f1, random_seed, time.ctime()]
    save_results_csv("results/results.csv", results, header=0)
    
    results = [step, roc_auc, prc_auc, precision, recall, f1, random_seed]
    save_results_csv(fname, results, header=2)

def save_results_csv(fname, results, header=0):
    """Saves results in csv file
    Args:
        fname (str): name of the file
        results (list): list of prec, rec, F1, rds
    """

    new_rows = []
    if not os.path.isfile(fname):
        args = fname.split('/')[:-1]
        directory = os.path.join(*args)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(fname, 'wt') as f:
            writer = csv.writer(f)
            if header == 0:
                writer.writerows(
                    [['Model', 'Dataset', 'AnomalyType', 'AnomalyProportion', 'Method', 'Weight', 'Label', 
                      'Step', 'AUROC', 'AUPRC', 'AUPRG', 'Precision', 'Recall',
                      'F1 score', 'Random Seed', 'Date']])

    with open(fname, 'at') as f:
        # Overwrite the old file with the modified rows
        writer = csv.writer(f)
        new_rows.append(results)  # add the modified rows
        writer.writerows(new_rows)

def heatmap(data, name=None, save=False):

    fig = plt.figure()
    ax = sns.heatmap(data, cmap="YlGnBu")
    fig.add_subplot(ax)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if save:
        args = name.split('/')[:-1]
        directory = os.path.join(*args)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig('{}.png'.format(name))
    return data

def save_scores(scores, labels, model, dataset, method):
    directory = 'scores/{}/{}/{}/'.format(dataset, model, method)
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_results_csv(directory + 'scores_reconstructions.csv', scores, header=4)
    save_results_csv(directory + 'scores_reconstructions.csv', labels, header=4)


