from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix, hstack, vstack
from tqdm import trange

import graph

import numpy as np

def prompt(first=False):
    fixed_nodes = None
    if first:
        fixed_nodes = list(map(
            int,
            input('Nodes to fix into subgraph '
                  '(integers separated by spaces):  ').split(' ')
        ))
        print()

    nodes = list(map(
        int,
        input('Nodes to compute recommendations for '
              '(integers separated by spaces,'
              ' must be a subset of fixed nodes):  ').split(' ')
    ))
    print()
    weights = input('Respective node weights for recommendations '
                    '(leave blank for unweighted):  ').split(' ')
    print()
    k = int(input('Number of top similar nodes to recommend:  '))
    print()
    if weights == ['']:
        weights = None
    else:
        weights = list(map(int, weights))
    return nodes, weights, k, fixed_nodes

'''Access upper triangle of matrix to compute neighbors of node.'''
def compute_neighbors(adj_matrix, node):
    return hstack((adj_matrix[:node, node].transpose(),
                   adj_matrix[node, node:]),
                   format='csr')

'''Return indices for top k elements of array.'''
def top_k(arr, k):
    top_k = np.argpartition(arr, -k)[-k:]
    top_k = top_k[np.argsort(arr[top_k])][::-1]
    return top_k

if __name__ == '__main__':
    nodes, weights, k, fixed_nodes = prompt(first=True)
    # load graph vertices and edges (takes a while for full graph)
    wg = graph.WikiGraph(fixed_nodes=fixed_nodes)
    # update node ids with subgraph ids
    nodes = [wg.selected_ids[id_] for id_ in nodes]
    adj_matrix = wg.adj_matrix
    N = adj_matrix.shape[0]
    print(f'Graph contains {N} vertices.')

    while True:
        # compute set of neighbors for each node in given nodes.
        set_neighbors = vstack(
            [compute_neighbors(adj_matrix, node) for node in nodes],
            format='csr'
        )
        # number of nodes being considered for recommendations
        n = len(nodes)

        # compute similarity scores for every node
        jaccard = np.zeros((n, N))
        simpson = np.zeros((n, N))

        for i in trange(N, desc='computing node similarities'):
            if i in nodes:
                # leave similarity at 0 for nodes in starting set
                continue
            neighbors = compute_neighbors(adj_matrix, i)
            for j in range(n):
                # need to convert 1x1 sparse matrix => 1x1 np.array => scalar
                jaccard_numerator = np.asscalar(
                    neighbors[0, :].dot(
                        set_neighbors[j, :].transpose()
                    ).A
                )
                # union is sum of cardinalities minus intersection
                start_node_degree = np.sum(set_neighbors[j, :])
                current_node_degree = np.sum(neighbors[0, :])
                jaccard_denominator = (
                    start_node_degree + current_node_degree - jaccard_numerator
                )
                if jaccard_numerator == 0 or jaccard_denominator == 0:
                    jaccard[j, i] = 0
                else:
                    jaccard[j, i] = jaccard_numerator / jaccard_denominator

                # both numerators are size of intersection of neighbors
                simpson_numerator = jaccard_numerator
                simpson_denominator = min(
                        start_node_degree, current_node_degree
                )
                if simpson_numerator == 0 or simpson_denominator == 0:
                    simpson[j, i] = 0
                else:
                    simpson[j, i] = simpson_numerator / simpson_denominator

        # prune nodes that are high similarity only because of one node
        # in the starting set
        for i in range(N):
            if np.count_nonzero(jaccard[:, i]) == 1:
                jaccard[:, i] = np.zeros(n)
            if np.count_nonzero(simpson[:, i]) == 1:
                simpson[:, i] = np.zeros(n)

        mean_jaccard = np.average(jaccard, axis=0, weights=weights)
        mean_simpson = np.average(simpson, axis=0, weights=weights)
        # l2_jaccard = np.linalg.norm(jaccard, axis=0)
        # l2_simpson = np.linalg.norm(simpson, axis=0)
        # argpartition returns top k indices in unsorted order, so sort after
        top_k_jaccard = top_k(mean_jaccard, k)
        top_k_simpson = top_k(mean_simpson, k)
        for i in top_k_jaccard:
            print(f'{wg.id_name_map[i]} has mean Jaccard similarity '
                  f'{mean_jaccard[i]} relative to the given articles.')
        for i in top_k_simpson:
            print(f'{wg.id_name_map[i]} has mean overlap coefficient '
                  f'{mean_simpson[i]} relative to the given articles.')
        print('Next set of inputs (ctrl-c to stop at any point):')
        nodes, weights, k, _ = prompt()
        nodes = [wg.selected_ids[id_] for id_ in nodes]
