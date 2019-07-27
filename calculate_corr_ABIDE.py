# Author: Xiuchao.Sui@gmail.com
# Editted: Ang Jun Liang
"""
Extract Timeseries from 264 sphere nodes of Power Atlas
Radius:  2.5 (5mm radius) -> in accord with (Power et al. 2011)

Input: 
    .nii.gz files - timeseries data
    .txt file - MNI coordinates for Power Atlas

Output:
    .npy file - rsfMRI data
"""

import pdb
import os
import re
import numpy as np
import itertools
import nibabel as nib
from multiprocessing import Pool
from nibabel.affines import apply_affine
import numpy.linalg as npl

def convertMNItoIJK(affine, mni_coord): 
    ijk = apply_affine(npl.inv(affine), mni_coord)
    ijk = np.rint(ijk)
    ijk_int = ijk.astype(int)
    return ijk_int

def setParameters():
    config = {'R': 2.5}
    print ("Sphere radius is %.1f mm" %(config['R']))
    return config

def isValidVoxel(matrix, v):
    if np.count_nonzero( (v < 0) | (v >= matrix.shape[:3]) ) > 0:
        return False
    if np.count_nonzero(matrix[tuple(v)]) == 0: # the timeseries contains all zeroes
        return False
    return True

def genBall(R):
    R2 = int(R)
    dv_inBall = []
    var_range = range( -R2, R2 + 1 )
    for dx, dy, dz in itertools.product(var_range, var_range, var_range):
        if dx*dx + dy*dy + dz*dz <= R*R:
            dv_inBall.append( [ dx, dy, dz ] )    
    return np.array(dv_inBall)
           
def getValidCoordsInBall(matrix, v, dv_inBall):
    coords_inBall = dv_inBall + v
    validCoordsInBall = []
    for v2 in coords_inBall:
        if isValidVoxel(matrix, v2):
            validCoordsInBall.append(v2)
    return np.array(validCoordsInBall)

def meanTS(matrix, ROI_voxs):
    ts_voxs = np.copy( matrix[ROI_voxs[:, 0], ROI_voxs[:, 1], ROI_voxs[:, 2]] )
    meanTimeseries = np.mean( ts_voxs, axis=0 )
    return meanTimeseries 

def saveResults(config, meanTimeseries, subject):
    pearsonCorr = np.corrcoef(meanTimeseries)
    np.fill_diagonal(pearsonCorr, 0)
    corrNumpyFile =  os.path.join(save_folder, subject + '_power.npy')

    try:
        os.mkdir(os.path.dirname(corrNumpyFile))
    except:
        pass
    np.save(corrNumpyFile, pearsonCorr)
    
def PowerSphere(params):
    img_file, power_coords_mni, subject = params
    

    print ("loading file from '%s'..." % (img_file))
    img = nib.load(img_file)
    affine = img.affine
    matrix = img.get_data()

    config = setParameters()
    config['affine'] = affine
    config['img_file'] = img_file

    K = config['K'] = len(power_coords_mni)
    dv_inBall = genBall(config['R'])

    # initialize 
    ROIs_voxels= [[]for k in range (K)]
    meanTimeseries  = np.zeros((K, matrix.shape[3]))
    power_coords_ijk = np.zeros((K, 3))
    power_coords_ijk = power_coords_ijk.astype(int)
        
    for k in range(K):
        power_coords_ijk[k] = convertMNItoIJK(config['affine'], power_coords_mni[k])
        ROIs_voxels[k] = getValidCoordsInBall(matrix, power_coords_ijk[k], dv_inBall)
        if len(ROIs_voxels[k]) != 0:
            meanTimeseries[k] = meanTS(matrix, ROIs_voxels[k])   

    saveResults(config, meanTimeseries, subject) 

    return True

# ---------------------------- Loop Subjects ----------------------------

if __name__ == '__main__':
    root = './Outputs/cpac/filt_global/func_preproc/' # where data is downloaded with download_abide_preproc.py
    save_folder = 'Processed_ABIDE/'
    subject_list = os.listdir(root)
    power_filename = 'PowerVoxMNI.txt'
    power_coords = np.genfromtxt(power_filename)
    K = len(power_coords)

    p = Pool(10)
    params = []
    for subject in subject_list:
        subject = subject[0:-20]

        if os.path.isdir(root):
            img_file = os.path.join(root, subject + '_func_preproc.nii.gz') # timeseries data
            try:
                os.mkdir(save_dir)
            except:
                pass

            if os.path.isfile(img_file):
                print ('------ subject %s file %s-----' %(subject, img_file))
                params.append(tuple([img_file, power_coords, subject]))

    p.map(PowerSphere, params)
    p.close() # no more tasks
    p.join()  # wrap up current tasks
