from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm

import numpy as np


NUM_ARTICLES = 1791489
SUBGRAPH_PERCENT = 5
NUM_NODES = SUBGRAPH_PERCENT * (NUM_ARTICLES // 100)
NUM_LINKS = 28511807

class WikiGraph:
    def __init__(self, undirected=True, fixed_nodes=[]):
        # first, grab all the neighbors for each fixed node and fix
        # them into the subgraph as well
        fixed_neighbors = []
        with open('wiki-topcats.txt') as f:
            links = (tuple(map(int, line.split(' '))) for line in f)
            for src_id, dst_id in tqdm(
                    links,
                    desc='constructing neighborhoods',
                    total=NUM_LINKS
            ):
                if src_id in fixed_nodes and dst_id not in fixed_nodes:
                    fixed_neighbors.append(dst_id)
                if dst_id in fixed_nodes and src_id not in fixed_nodes:
                    fixed_neighbors.append(src_id)
        fixed_nodes += fixed_neighbors

        # select the rest of the nodes in the graph randomly so that
        # we end up with SUBGRAPH_PERCENT of the nodes in the graph
        selected_nodes = np.concatenate(
            [np.array(fixed_nodes),
             np.random.choice(NUM_ARTICLES, NUM_NODES - len(fixed_nodes))]
        )

        # remember the new indices for each id
        selected_ids = {}
        id_ctr = 0

        # load article names
        id_name_map = {}
        with open('wiki-topcats-page-names.txt') as f:
            for line in tqdm(f, desc='article names', total=NUM_ARTICLES):
                # names can have spaces, so split only once
                id_, name = line.split(' ', maxsplit=1)
                id_ = int(id_)
                if id_ not in selected_nodes:
                    continue

                # update map of old id to new id in subgraph
                new_id = id_ctr
                selected_ids[id_] = new_id
                id_ctr += 1

                # remember name with new id
                name = name.strip()
                id_name_map[new_id] = name
        # not all indices from 0 - NUM_ARTICLES exist, apparently, so
        # keep track of how many were actually found
        num_nodes = id_ctr
        adj_matrix = lil_matrix((num_nodes, num_nodes),
                                dtype=np.dtype(np.int8))

        # load links
        with open('wiki-topcats.txt') as f:
            links = (tuple(map(int, line.split(' '))) for line in f)
            for src_id, dst_id in tqdm(links, desc='links', total=NUM_LINKS):
                if (src_id not in selected_nodes
                        or dst_id not in selected_nodes):
                    continue
                # at this point we know this edge is in the subgraph, so
                # translate to new ids
                src_id, dst_id = selected_ids[src_id], selected_ids[dst_id]
                if undirected:
                    # store only upper triangle
                    if src_id <= dst_id:
                        adj_matrix[src_id, dst_id] = 1
                    else:
                        adj_matrix[dst_id, src_id] = 1
                else:
                    adj_matrix[src_id, dst_id] = 1

        self.selected_ids = selected_ids
        self.id_name_map = id_name_map
        self.links = links
        self.adj_matrix = csr_matrix(adj_matrix)
