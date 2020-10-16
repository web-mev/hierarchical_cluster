import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input',
        required=True,
        dest = 'input_matrix',
        help='The input matrix'
    )

    parser.add_argument('-d', '--dist', \
        dest = 'dist_metric',
        required=False,
        default = 'euclidean',
        help=('The distance metric to use for clustering.')
    )

    parser.add_argument('-l', '--linkage', \
        dest = 'linkage',
        required=False,
        default = 'ward',
        help=('The linkage criterion to use for clustering.')
    )

    parser.add_argument('-c', '--cluster', \
        dest = 'cluster_dim',
        required=False,
        default = 'both',
        choices = ['both','observations','features'],
        help=('Whether to cluster observations, features, or both.')
    )

    args = parser.parse_args()
    return args

def create_linkage_matrix(clustering):
    '''
    Given the results from the clustering, create a 
    linkage matrix which can be used to create a dendrogram
    '''
    counts = np.zeros(clustering.children_.shape[0])
    n_samples = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    return np.column_stack([
        clustering.children_, 
        clustering.distances_,
        counts]).astype(float)

def cluster(df, dist_metric, linkage):
    '''
    Perform the clustering and return the cluster object.

    The first arg is a matrix or dataframe. It is assumed that the dataframe
    is oriented in the correct manner according to sklearn/scipy, which
    clusters on the rows of the matrix.

    Thus, if the matrix is (n,m) it will cluster the `n` row-wise objects, regardless
    of whether they represent observations or features
    '''

    # Due to a constraint with scikit-learn, need to massage the args a bit.
    # For instance, even with a euclidean-calculated distance matrix, the cluster
    # method still errors since it NEEDS affinity='euclidean' if linkage='ward'

    if linkage == 'ward':
        mtx = df
        affinity = 'euclidean'
    else:
        # calculate the distance matrix. By default it's given as a vector,
        # but the `squareform` function turns it into the conventional square
        # distance matrix.
        try:
            mtx = squareform(pdist(df, dist_metric))
        except Exception as ex:
            sys.stderr.write('Failed when calculating the distance matrix.'
                ' Reason: {ex}'.format(ex=ex)
            )
            sys.exit(1)
        affinity = 'precomputed'

    # Now perform the clustering. We used the pdist function above to expand
    # beyond the typical offerings for the distance arg
    try:
        clustering = AgglomerativeClustering(
            affinity = affinity,
            compute_full_tree = True, # required when distance_threshold is non-None
            linkage = linkage,
            distance_threshold=0, # full tree
            n_clusters=None # required when distance_threshold is non-None
        ).fit(mtx)
        return clustering
    except Exception as ex:
        sys.stderr.write('Failed when clustering.'
            ' Reason: {ex}'.format(ex=ex)
        )
        sys.exit(1)
    
if __name__ == '__main__':
    
    args = parse_args()
    working_dir = os.path.dirname(args.input_matrix)

    # read the matrix. This is, by our convention, (num features, num samples)
    df = pd.read_table(args.input_matrix, index_col=0)

    feature_linkage_output, observation_linkage_output = None, None
    if (args.cluster_dim == 'features') or (args.cluster_dim == 'both'):
        feature_clustering = cluster(df, args.dist_metric, args.linkage)
        feature_linkage_mtx = create_linkage_matrix(feature_clustering)
        feature_linkage_output = 'hcl_features.tsv'
        np.savetxt(
            feature_linkage_output, 
            feature_linkage_mtx, 
            delimiter='\t', 
            fmt=['%d','%d','%.3f','%d']
        )
    if (args.cluster_dim == 'observations') or (args.cluster_dim == 'both'):
        observation_clustering = cluster(df.T, args.dist_metric, args.linkage)
        observation_linkage_mtx = create_linkage_matrix(observation_clustering)
        observation_linkage_output = 'hcl_observations.tsv'
        np.savetxt(
            observation_linkage_output, 
            observation_linkage_mtx, 
            delimiter='\t', 
            fmt=['%d','%d','%.3f','%d']
        )

    outputs = {
        'hcl_features': feature_linkage_output,
        'hcl_observations': observation_linkage_output
    }
    json.dump(outputs, open(os.path.join(working_dir, 'outputs.json'), 'w'))
