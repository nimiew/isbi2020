import deeplift 
import numpy as np 
from deeplift.conversion import kerasapi_conversion as kc
from sklearn.model_selection import train_test_split
import os
import keras
from scipy import stats
import matplotlib.pyplot as plt

def reverse_rearrangement(matrix, permutation):
	# the permutation tells what is the node at index i. We wish to make node at index i to be i. 
	new_x = np.zeros((permutation.shape[0], permutation.shape[0]))

	for old_node_1, new_node_1 in enumerate(permutation):
		for old_node_2, new_node_2 in enumerate(permutation):
			new_x[new_node_1, new_node_2] = matrix[old_node_1, old_node_2]
		
	return new_x

def get_reference(mode, reference_label, data, data_labels, each_sample_index = 0):
	reference = np.zeros(data[0].shape)
	num_reference_samples = np.sum(data_labels == reference_label) 

	average_reference, norm_reference, modal_reference = np.zeros(data[0].shape), np.zeros(data[0].shape), np.zeros(data[0].shape)

	print('mode is', mode)
	
	if mode == 'average' or mode == 'best':
		for index, x in enumerate(data_labels):
			if x == reference_label:
				reference += data[index]
		reference = reference/num_reference_samples
		
		average_reference = np.copy(reference)

	if mode == 'each':
		for index, sample in enumerate(data[data_labels == reference_label]):
			if index == each_sample_index:
				reference = sample
	
	if mode == 'representative' or mode == 'best':
		similarity_scores = dict()
		for index_1, sample_1 in enumerate(data[data_labels == reference_label]):
			similarity_scores[index_1] = 0
			for _, sample_2 in enumerate(data[data_labels == reference_label]):
		 		similarity_scores[index_1] += np.linalg.norm(sample_1 - sample_2)

		reference_index = min(similarity_scores, key=similarity_scores.get)
		reference = data[data_labels == reference_label][reference_index]

		norm_reference = np.copy(reference)

	if mode == 'modal' or mode == 'best':
		reference_data = data[data_labels == reference_label]
		for i in range(reference_data.shape[1]):
			vals= np.around(reference_data[:, i], decimals=2)
			reference[i], _  = stats.mode(vals, nan_policy = 'omit')

		modal_reference = np.copy(reference)

	if mode == 'best':
		similarity_scores = dict()
		references_dict = {'average': average_reference, 'norm': norm_reference, 'modal': modal_reference}
		
		for ref in references_dict.keys():
			similarity_scores[ref] = 0
			for index, sample in enumerate(data[data_labels == reference_label]):
		 		similarity_scores[ref] += np.linalg.norm(references_dict[ref] - sample)
		 		print (ref, index, similarity_scores[ref], np.linalg.norm(references_dict[ref] - sample))

		best_reference = min(similarity_scores, key=similarity_scores.get)
		reference = references_dict[best_reference]

		print (mode, similarity_scores, best_reference)
    
	num_copies = data.shape[0] - np.sum(data_labels == reference_label) 
	rep_reference = []
	for _ in range(num_copies):
		rep_reference.append(reference)

	return rep_reference

def get_padded_data(data, mapping):
	padded_data = np.zeros(len(mapping))
	for orig_index in mapping.keys():
		new_index = mapping[orig_index]
		if new_index != -1:
			padded_data[orig_index] = data[new_index]

	return padded_data

def get_masked_data(data, mask): # data shape is (num samples, num features)
	masked_data = []
	mapping = dict() # key is the original index, value is the new index
	new_index = 0
	for orig_index, value in enumerate(mask):
		if value == 1:
			mapping[orig_index] = new_index
			new_index += 1
		else:
			mapping[orig_index] = -1
	
	for sample in data:
		masked_sample = np.zeros(int(np.sum(mask)))
		for orig_index in mapping.keys():
			new_index = mapping[orig_index]
			if new_index != -1:
				masked_sample[new_index] = sample[orig_index]

		masked_data.append(masked_sample)

	return np.array(masked_data), mapping

def flatten_data(matrix):
	matrix_lower_triangular = matrix[np.triu_indices(np.shape(matrix)[0], 1)]
	matrix_flat = np.ravel(matrix_lower_triangular, order="C")
	return matrix_flat

def get_crucial_features_abs(matrix, percentage_cutoff):
	flat_matrix = flatten_data(matrix)

	abs_sensitivity = np.absolute(flat_matrix) 
	abs_sensitivity_sorted = np.sort(abs_sensitivity)

	threshold = abs_sensitivity_sorted[int((1-percentage_cutoff)*len(flat_matrix))]

	selected_positive_features = np.where(flat_matrix > threshold)[0]
	selected_negative_features = np.where(flat_matrix < -threshold)[0]

	selected_features = np.zeros(flat_matrix.shape)
	selected_features[selected_positive_features] = 1
	selected_features[selected_negative_features] = 1

	reshape_selected_features_matrix = np.zeros(matrix.shape)
	reshape_selected_features_matrix[np.triu_indices(matrix.shape[0], 1)] = selected_features
	reshape_selected_features_matrix_T = reshape_selected_features_matrix.T
	reshape_selected_features_matrix = reshape_selected_features_matrix + reshape_selected_features_matrix_T - np.diag(np.diag(reshape_selected_features_matrix))
	return reshape_selected_features_matrix
	

def compute_deeplift_scores(dataset, X, Y, keras_model_file, reference_label, non_reference_label, base_neuron_label, mask, gpu_id, threshold, percentage_cutoff):
	os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu_id)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	X_masked, mapping = get_masked_data(X, mask)
	Y = np.argmax(Y, axis=1)
	
	task_id = base_neuron_label
	
	find_scores_layer_idx = 0
	
	mode = 'average'

	reference = get_reference(mode, reference_label, X_masked, Y)
	deeplift_model = kc.convert_model_from_saved_files(keras_model_file, nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)

	deeplift_contribs_func = deeplift_model.get_target_contribs_func(
	                            find_scores_layer_idx=find_scores_layer_idx,
	                            target_layer_idx=-2)

	scores = np.array(deeplift_contribs_func(task_idx=task_id,
											 input_references_list = reference,
	                                         input_data_list=[X_masked[Y == non_reference_label]],
	                                         batch_size=10,
	                                         progress_update=10))

	sum_scores = np.zeros(X_masked.shape[1])
	
	for score in scores:
		sum_scores += score

	padded_sum_scores = get_padded_data(sum_scores, mapping)
	print("Reshaping scores ")
	full_matrix = np.zeros((264, 264))
	full_matrix[np.triu_indices(264, 1)] = padded_sum_scores
	full_matrix_T = full_matrix.T
	full_matrix = full_matrix + full_matrix_T - np.diag(np.diag(full_matrix_T))

	if not os.path.isdir('./important_features/'):
	    print("Folder that will store the results cannot be found.")
	    print("Creating the results folder in " + './important_features/')
	    os.makedirs('./important_features/')

	np.savetxt('./important_features/' + dataset + '_scores_deeplift_reduced_r_' + str(threshold) + '_t_' + str(percentage_cutoff) + '.csv', np.transpose(np.array(scores)), delimiter= ',')

	print('Writing reshaped scores')
	np.savetxt('./important_features/' + dataset + '_scores_reshaped_reduced_r_' + str(threshold) + '_t_' + str(percentage_cutoff) + '.csv', full_matrix, delimiter=",")

	selected_features_matrix = get_crucial_features_abs(full_matrix, percentage_cutoff)
	np.fill_diagonal(selected_features_matrix, 0)

	selected_features_file = './important_features/' + dataset + '_deeplift_features_nodes_r_' + str(threshold) + '_t_' + str(percentage_cutoff) + '.csv'
	np.savetxt(selected_features_file, selected_features_matrix)
	
	return selected_features_file, flatten_data(selected_features_matrix)

