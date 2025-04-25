import numpy as np


def stoer_wagner(weights):
    """
    Computes the global minimum cut for an undirected weighted graph
    using the Stoer-Wagner algorithm.

    Parameters:
      weights: a 2D numpy array of nonnegative numbers, where weights[i, j]
               is the weight of the edge between nodes i and j. It must be
               symmetric with zeros on the diagonal.

    Returns:
      A tuple (set_A, set_B, cut_weight) where:
         - set_A and set_B are two disjoint sets partitioning the nodes {0,...,n-1}.
         - cut_weight is the sum of weights of edges crossing between set_A and set_B.
    """
    # Ensure weights is a NumPy array (if not already)
    weights = np.array(weights)
    n = weights.shape[0]
    if n == 0:
        return set(), set(), 0

    # List of current vertex indices (merging will remove vertices)
    vertices = list(range(n))
    # Keep track of merged sets for each vertex
    sets = {i: {i} for i in vertices}

    best_cut_weight = float("inf")
    best_cut = None

    while len(vertices) > 1:
        used = [False] * len(vertices)
        # Array to keep track of the connectivity weights for the phase.
        weights_to_set = [0] * len(vertices)
        prev = 0

        for i in range(len(vertices)):
            # Select the vertex with the maximum weight that hasn't been used.
            sel = None
            for j in range(len(vertices)):
                if not used[j] and (
                    sel is None or weights_to_set[j] > weights_to_set[sel]
                ):
                    sel = j

            used[sel] = True

            # If this is the last vertex in this phase, we have a cut.
            if i == len(vertices) - 1:
                if weights_to_set[sel] < best_cut_weight:
                    best_cut_weight = weights_to_set[sel]
                    best_cut = sets[vertices[sel]].copy()  # one partition of the cut

                # Merge the last two vertices: vertices[prev] and vertices[sel]
                merged_vertex = vertices[prev]
                merging_vertex = vertices[sel]
                sets[merged_vertex].update(sets[merging_vertex])

                # Update the weights for the merged vertex.
                for j in range(len(vertices)):
                    weights[merged_vertex, vertices[j]] += weights[
                        merging_vertex, vertices[j]
                    ]
                    weights[vertices[j], merged_vertex] = weights[
                        merged_vertex, vertices[j]
                    ]
                vertices.pop(sel)
                break

            prev = sel
            # Update the connectivity weights for the remaining vertices.
            for j in range(len(vertices)):
                if not used[j]:
                    weights_to_set[j] += weights[vertices[sel], vertices[j]]

    full_set = set(range(n))
    return best_cut, full_set - best_cut, best_cut_weight


# Example usage:
if __name__ == "__main__":
    # Example graph: 4 nodes with a symmetric weight matrix.
    weights = np.array([[0, 3, 1, 4], [3, 0, 2, 1], [1, 2, 0, 5], [4, 1, 5, 0]])

    set_A, set_B, cut_weight = stoer_wagner(weights)
    print("Partition A:", set_A)
    print("Partition B:", set_B)
    print("Cut weight (sum of edges between A and B):", cut_weight)
