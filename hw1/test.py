# Enter your code here. Read input from STDIN. Print output to STDOUT
from typing import Iterable, Callable

import numpy as np
import sys

# Computes the distances between an NxD matrix and an MxD matrix and returns an NxM matrix of distances.
distanceFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def distance(d1, d2):
    total = 0
    for i in range(len(d1)):
        total += (d1[i] - d2[i]) ** 2
    return total ** 0.5

def euclidian(q,r):


class KNN:
    # Fill in your KNN implementation here.
    def __init__(self, references, distanceFn=distance, K=0):
        self.references = references
        self.K = K

    @property
    def nearest_k(self) -> np.ndarray:
        """Returns a numpy array of size [K x D] where K is the number of nearest
        neighbors we want to keep and D is the dimensionality. The queries must
        be sorted from smallest distance to largest distance (to any reference point).
        """
        arr = np.arange(0)

        return arr

    def observe(self, images):
        distances = np.arange(0)
        for re in self.references:
            for image in images:
                np.concatenate((distances, self.distanceFn(image, re)))


def get_nearest_neighbors(
        references: np.ndarray,  # Shape=[N, D]
        distance_function_name: str,
        k: int,
        queries: Iterable[np.ndarray]
    ) -> np.ndarray:
    assert distance_function_name in ('L2', 'cityblock')

    # Create the distance function which takes the query matrix (B, D) and the reference
    # matrix (N, D) and produces a matrix of distances between all pairs of size (B, N).
    #
    # my_distance_fn = ...
    #
    # Your solution should look something like this:
    # knn = KNN(references, distanceFn=my_distance_fn, K)
    # for query in queries:
    #    # The shape of query = [B, D].
    #    knn.observe(query)
    # return knn.nearest_k
    knn = KNN(references, distanceFn=distance, k)


import json, os, sys

if __name__ == "__main__":
    config = json.load(sys.stdin)
    reference_points = np.array(config["reference_points"])
    distance_function_name = config["distance_function"]
    k = config["k"]
    queries = [np.array(query) for query in config["queries"]]
    actual_nearest_k = get_nearest_neighbors(reference_points, distance_function_name, k, queries)
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    fptr.write(str(actual_nearest_k.astype(float).tolist()))
    fptr.close()



# {
#      "reference_points": [[-6, 2], [-4, 4]],
#      "distance_function": "L2",
#      "k": 2,
#      "queries": [[[0, -1], [0, -2], [-5, 3], [0, 10]], [[0, 1], [4, 3.8], [0, 9], [-4, 6]]]
# }
#
# [[-5.0, 3.0], [-4.0, 6.0]]