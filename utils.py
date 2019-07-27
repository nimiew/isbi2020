import os 
import numpy as np 
from random import randint

SOURCE_DIRECTORY = '../../../data/' 

def prepare_dataset_abide_matrices_masked(mask):

    num_remaining_features = np.count_nonzero(np.sum(mask, axis = 0), axis=None) 
    num_features = (num_remaining_features, num_remaining_features)
    non_zero_rows = np.where(np.sum(mask, axis = 0) > 0)[0]

    print('num_features: ' + str(num_features))

    # control
    all_matrices_cn = []

    if ".DS_Store" in os.listdir(SOURCE_DIRECTORY + "normal/"):
        os.remove(SOURCE_DIRECTORY + "/CN/.DS_Store")
        print(".DS_Store removed from " + SOURCE_DIRECTORY + "normal/")

    for i, file_or_dir in enumerate(os.listdir(SOURCE_DIRECTORY + "normal/")):
        if ".DS_Store" not in file_or_dir:
            all_matrices_cn.append(np.load(SOURCE_DIRECTORY + "normal/" + file_or_dir))


    for i, matrix in enumerate(all_matrices_cn):
        matrix = np.nan_to_num(matrix)
        masked_matrix = np.multiply(matrix, mask)
        reduced_matrix = masked_matrix[np.ix_(non_zero_rows, non_zero_rows)]
        all_matrices_cn[i] = reduced_matrix

    all_matrices_diseased = []

    if ".DS_Store" in os.listdir(SOURCE_DIRECTORY + "diseased/"):
        os.remove(SOURCE_DIRECTORY + "/MCI/.DS_Store")
        print(".DS_Store removed from " + SOURCE_DIRECTORY + "diseased/")

    for i, file_or_dir in enumerate(os.listdir(SOURCE_DIRECTORY + "diseased/")):
        if ".DS_Store" not in file_or_dir:
            all_matrices_diseased.append(np.load(SOURCE_DIRECTORY + "diseased/" + file_or_dir))


    for i, matrix in enumerate(all_matrices_diseased):
        matrix = np.nan_to_num(matrix)
        masked_matrix = np.multiply(matrix, mask)
        reduced_matrix = masked_matrix[np.ix_(non_zero_rows, non_zero_rows)]
        all_matrices_diseased[i] = reduced_matrix


    all_matrices = np.empty((len(all_matrices_cn) + len(all_matrices_diseased), num_features[0], num_features[1]))

    for i, matrix in enumerate(all_matrices):  
        if i < len(os.listdir(SOURCE_DIRECTORY + 'normal')): 
            all_matrices[i] = all_matrices_cn[i]
        elif i < len(os.listdir(SOURCE_DIRECTORY + 'normal')) + len(os.listdir(SOURCE_DIRECTORY + 'diseased')):
            all_matrices[i] = all_matrices_diseased[i - (len(os.listdir(SOURCE_DIRECTORY + 'normal')))]
        else: 
            print("There are more matrices than expected!")

    # labels
    label_cn = [0 for i in range(len(all_matrices_cn))]
    label_diseased = [1 for i in range(len(all_matrices_diseased))]

    all_labels = np.array(label_cn + label_diseased) 

    Y = np.zeros((all_matrices.shape[0], 2))
    for i in range(all_labels.shape[0]):
        Y[i, all_labels[i]] = 1  # 1-hot vectors

    return (all_matrices, Y)
