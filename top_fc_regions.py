"""
Requires power_labels.csv and squared_scores_reshaped.csv in the same directory
Prints the regions associated with top k functional connectivities and saves brain figure with FCs
Usage: python3 deeplift_analysis.py -k
"""

import numpy as np
import argparse
from nilearn import plotting

def get_names(labels, region):
    return list(labels[region][1:])

def k_largest(scores, k):
    idx = np.argsort(scores.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, scores.shape)).tolist()

def unique_regions(scores, k):
    regions = []
    for list_ in k_largest(scores, k):
        regions += list_
    return list(set(regions))

def unique_names(scores, labels, k):
    names = []
    regions = unique_regions(scores, k)
    for region in regions:
        names += get_names(labels, region)
    return list(set(names))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prints the regions associated with top k functional connectivities')
    parser.add_argument('-k', '--top', type=int, required=True, help="top k functional connectivities")
    parser.add_argument('-s', '--sort', action='store_true', default=False, help="sort?")
    args = parser.parse_args()

    labels = np.genfromtxt('power_labels.csv', delimiter=',', dtype='str')
    scores = np.genfromtxt('squared_scores_reshaped.csv', delimiter=',')
    coords = np.array([i.split("\t")[1:] for i in np.genfromtxt('power_coords.txt', delimiter=' ', dtype='str')])
    
    unique_names_list = unique_names(scores, labels, args.top * 2)
    if args.sort: unique_names_list.sort()
    for region in unique_names_list:
        print(region)

    display = plotting.plot_connectome(adjacency_matrix=scores,
                             node_coords=coords,
                             node_size=0.0,
                             edge_threshold="99%",
                             edge_cmap = "YlOrBr",
                             )
    display.savefig('pretty_brain.png')