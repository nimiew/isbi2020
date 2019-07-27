import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, LocallyConnected2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.engine.topology import Layer
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import numpy as np
import os
from rfe_sensitivity_flat_ABIDE import compute_deeplift_scores
from sklearn.model_selection import train_test_split
from utils import prepare_dataset_abide_matrices_masked 
import tensorflow as tf
from matplotlib import pyplot as plt
from pylab import savefig
from sklearn import metrics
import sys

class Hadamard(Layer):

    def __init__(self, **kwargs):
        self.name = 'Hadamard'
        self.trainable = True
        super(Hadamard, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape= input_shape[1:],
                                      initializer= keras.initializers.RandomUniform(minval=0.05, maxval=1, seed=None),
                                      trainable=True)
        super(Hadamard, self).build(input_shape)

    def call(self, x):
        print(x.shape, self.kernel.shape)
        print ('self.kernel: ', self.kernel)
        out = np.multiply(x,self.kernel)
        print ('x * self.kernel', out)
        print ('out.shape', out.shape)

        return out

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return input_shape

def modular_rearrangement_Graclus(X, perm):
    X_new = []
    N = X[0].shape[0]

    for x in X:
        new_x = np.zeros((perm.shape[0], perm.shape[0]))

        for new_node_1, old_node_1 in enumerate(perm):
            for new_node_2, old_node_2 in enumerate(perm):
                if old_node_1 < N  and old_node_2 < N :
                    new_x[new_node_1, new_node_2] = x[old_node_1, old_node_2]
                else: 
                    new_x[new_node_1, new_node_2] = 0
        
        X_new.append(new_x)        

    non_zero_rows = np.where(perm < N)[0]

    X_reduced = []

    for sample in X_new:
        reduced_matrix = sample[np.ix_(non_zero_rows, non_zero_rows)]
        X_reduced.append(reduced_matrix)

    return np.array(X_reduced)

def random_rearrangement(X, module_vector):
    num_nodes = X[0].shape[0]
    new_order = np.random.choice(num_nodes, size=num_nodes, replace=False)

    mapping = dict() # key is the row (represented by node index) in the original connectivity matrix and value is the corresponding row in connectivity rearranged matrix
    
    for node in range(num_nodes):
        mapping[node] = new_order[node]
        
    X_new = []

    for x in X:
        new_x = np.zeros(x.shape)
        for old_node_1 in mapping.keys():
            new_node_1 = mapping[old_node_1]
            for old_node_2 in mapping.keys():
                new_node_2 = mapping[old_node_2]
                new_x[new_node_1, new_node_2] = x[old_node_1, old_node_2]
    
        X_new.append(new_x)

    return np.array(X_new), new_order
    
def normalize_data(X, max_, min_):
    for i in range(X.shape[0]): 
        X[i] = (X[i] - min_)/(max_ - min_)

    return X

def standardize_data(X, mean_, std_):
    for i in range(X.shape[0]):
        X[i] = (X[i] - mean_)
    return X

def flatten_data(X):
    num_features = int((X.shape[1]) * (X.shape[1] - 1) * 0.5)
    X_flattened = np.empty((X.shape[0], num_features))

    for i, matrix in enumerate(X):
        matrix_lower_triangular = matrix[np.triu_indices(np.shape(matrix)[0],1)]
        X_flattened[i] = np.ravel(matrix_lower_triangular, order="C")

    return X_flattened

def funcNetFFN_1L(input_shape, dropout, batch_size):
    model = keras.models.Sequential()
    model.add(Dense(5, activation='relu', input_dim=input_shape))
    model.add(Dropout(rate=dropout))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model


def funcNetFFN_2L(input_shape, dropout, batch_size):
    model = keras.models.Sequential()
    model.add(Dense(50, activation='relu', input_dim=input_shape))
    model.add(Dropout(rate=dropout))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(rate=dropout))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model
    
def generate_compunded_neurons(multipler, number_of_values):
    neuron_list = []
    start = 1
    count = 0
    while(number_of_values):
        neuron_list.append(start * (multipler ** count))
        count += 1
        number_of_values -= 1

    return neuron_list

def custom_categorical_crossentropy(l1, layer_0_weights, from_logits=False):
    def orig_loss(y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred) + l1 * K.sum(K.abs(layer_0_weights))
    return orig_loss

class_subset = 'ABIDE'
mode = 'modular'
model_type = sys.argv[1] # funcNetFFN_1L funcNetFFN_2L
seeds = range(10, 20)
gpu_id = '3'

TARGET_DIRECTORY = './' + model_type + '/'

if not os.path.isdir(TARGET_DIRECTORY):
    print("Folder that will store the results cannot be found.")
    print("Creating the results folder in " + TARGET_DIRECTORY)
    os.makedirs(TARGET_DIRECTORY)

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= gpu_id
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

if not os.path.isdir(TARGET_DIRECTORY):
    print("Folder that will store the results cannot be found.")
    print("Creating the results folder in " + TARGET_DIRECTORY)
    os.makedirs(TARGET_DIRECTORY)


for SEED in seeds:
    mask = np.ones(34716)
    thresholds = np.array([1.0, 1.0]) # generate_compunded_neurons(0.9, 23) #np.linspace(1, 0.1, 10)
    X_prev, Y_prev = [], []
    for ix, i in enumerate(thresholds):
        ## Model parameters
        epochs = 100 # there's early stopping
        batch_size = 8
        learning_rate =0.0001
        decay = 0.001
        dropout = 0.1
        folds = 5
        np.random.seed(SEED)

        if ix == 0:
            (X,Y) = prepare_dataset_abide_matrices_masked(np.ones((264, 264)))
            if model_type == 'funcNetFFN_1L' or model_type == 'funcNetFFN_2L':
                X = flatten_data(X)

        else:
            threshold_1 = thresholds[ix-1]
            threshold_2 = thresholds[ix]
            prev_threshold = thresholds[ix-1]
            prev_best_model = TARGET_DIRECTORY + 'best_model_seed_' + str(SEED) + '_' + str(class_subset) + '_' + model_type + '_' + mode + '_' + str(prev_threshold) + '.h5'
            sensitivity_filename, mask = compute_deeplift_scores(class_subset + '_' + model_type + '_' + mode + '_SEED_' + str(SEED), X_prev, Y_prev, prev_best_model, 0, 1, 0, mask, gpu_id, threshold_1, threshold_2)
            (X,Y) = prepare_dataset_abide_matrices_masked(np.ones((264, 264)))
            X_flat = flatten_data(X)
            num_features = int(np.sum(mask))
            X = []
            for matrix in X_flat:
                masked_matrix = np.multiply(matrix, mask)
                X.append(masked_matrix[mask == 1])
            X = np.array(X)

            print ('Shape of input data', X.shape)
            print ('Number of features', num_features)

        idx = np.arange(len(X))
        np.random.shuffle(idx) # randomize index
        X, Y = X[idx], Y[idx]  # randomize/shuffle dataset
        
        input_shape = X[0].shape
        all_fold_accuracies = []
        
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        fold_count = 0
        model_filename_list = []
        train_indices, val_indices = [], []
        for train_index, val_index in skf.split(X, Y.argmax(1)):
            train_indices.append(train_index)
            val_indices.append(val_index)
            X_train, Y_train = X[train_index], Y[train_index]
            X_val, Y_val = X[val_index], Y[val_index]

            if model_type != 'funcNetFFN_1L' and model_type != 'funcNetFFN_2L':
                print("Invalid model type")
                exit()

            # Define model
            if model_type == 'funcNetFFN_1L':
                input_shape = X.shape[1]
                model = funcNetFFN_1L(input_shape, dropout, batch_size)
                loss_fn = "categorical_crossentropy"
            elif model_type == 'funcNetFFN_2L':
                input_shape = X.shape[1]
                model = funcNetFFN_2L(input_shape, dropout, batch_size)
                loss_fn = "categorical_crossentropy"

            Adam = optimizers.Adam(lr=learning_rate) #, decay=decay

            if loss_fn == 'custom':
                model.compile(loss= custom_categorical_crossentropy(0.001, model.layers[0].get_weights()[0]), optimizer=Adam, metrics=["accuracy"])
            else:
                model.compile(loss= "categorical_crossentropy", optimizer=Adam, metrics=["accuracy"])

            model_filename = TARGET_DIRECTORY + 'best_model_seed_' + str(SEED) + '_' +  str(class_subset) + '_' + model_type + '_' + mode + '_' + str(i) + '_fold_' + str(fold_count) + '.h5'
            model_filename_list.append(model_filename)
            callbacks = [ModelCheckpoint(filepath=model_filename, monitor='val_acc', save_best_only=True)]

            # Train model
            history = model.fit(X_train, Y_train, batch_size=batch_size, callbacks=callbacks, epochs=epochs, validation_data=(X_val, Y_val), verbose=1)
            score = model.evaluate(X_val, Y_val, batch_size=batch_size)
            
            best_val_acc = max(history.history['val_acc'])
            best_epoch = history.history['val_acc'].index(max(history.history['val_acc']))
            model.load_weights(model_filename)

            best_prediction = model.predict(X_val, batch_size=batch_size, verbose=1)
            MAE = metrics.mean_absolute_error(Y_val, best_prediction, sample_weight=None, multioutput='uniform_average')
            AUC = metrics.roc_auc_score(Y_val, best_prediction, average='macro', sample_weight=None, max_fpr=None)

            print(SEED, 'best accuracy original: ', best_val_acc, 'loaded: ', model.evaluate(X_val, Y_val, batch_size=batch_size), 'at epoch ', best_epoch, 'MAE ', MAE, 'AUC ', AUC)

            all_fold_accuracies.append(best_val_acc)

            with open(TARGET_DIRECTORY +  model_type + '_' + mode + '_kfold_training_logs_' + class_subset + '.csv', 'a') as out_stream:
                out_stream.write(str(SEED) + ', ' + str(i) + ', ' + str(fold_count) + ', ' + str(best_epoch) + ', ' + str(best_val_acc) + ', ' + str(MAE) + ', ' + str(AUC) + '\n')

            keras.backend.clear_session()
            fold_count += 1

        best_fold = all_fold_accuracies.index(max(all_fold_accuracies))
        best_model_name = model_filename_list[best_fold]
        print('Fold number ' + str(best_fold) + ' has the highest accuracy score of ' + str(max(all_fold_accuracies)))

        val_index = val_indices[best_fold]
        X_, Y_ = prepare_dataset_abide_matrices_masked(np.ones((264, 264)))

        if model_type == 'funcNetFFN_1L' or model_type == 'funcNetFFN_2L':
            X_ = flatten_data(X_)

        X_prev, Y_prev = X_[val_index], Y_[val_index]

        for model_file_fold in model_filename_list:
            if model_file_fold != best_model_name:
                os.remove(model_file_fold)
            else:
                filename = TARGET_DIRECTORY + 'best_model_seed_' + str(SEED) + '_' + str(class_subset) + '_' + model_type + '_' + mode + '_' + str(i) + '.h5'
                os.rename(model_file_fold, filename)
