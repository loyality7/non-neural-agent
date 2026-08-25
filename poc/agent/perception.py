"""Online clustering: turning a stream of noisy numeric vectors into a
small number of stable, reusable category labels, with no labels or
category count given in advance.

This is the perceptual-discovery step every prior experiment skipped by
being handed a clean color string directly. Nothing here is neural or
gradient-based: it is a simple, incremental nearest-centroid clusterer --
each new vector either joins its closest existing cluster (if close
enough) and nudges that cluster's running-mean centroid toward it, or
founds a new cluster if nothing is close enough. The number of clusters is
discovered, not specified.
"""

DEFAULT_DISTANCE_THRESHOLD = 0.5


def _distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


class OnlineClusterer:
    def __init__(self, distance_threshold=DEFAULT_DISTANCE_THRESHOLD):
        self.distance_threshold = distance_threshold
        self.centroids = []  # list of [mean_vector, count]

    def classify(self, vector):
        """Assigns `vector` to its nearest existing cluster if within the
        distance threshold (updating that cluster's centroid), otherwise
        founds a new cluster. Returns a cluster label (an integer index),
        used exactly like the color-string feature was used before.
        """
        best_idx, best_dist = None, float("inf")
        for i, (centroid, _) in enumerate(self.centroids):
            d = _distance(vector, centroid)
            if d < best_dist:
                best_idx, best_dist = i, d

        if best_idx is not None and best_dist <= self.distance_threshold:
            centroid, count = self.centroids[best_idx]
            new_count = count + 1
            new_centroid = tuple(
                c + (v - c) / new_count for c, v in zip(centroid, vector)
            )
            self.centroids[best_idx] = (new_centroid, new_count)
            return best_idx

        self.centroids.append((vector, 1))
        return len(self.centroids) - 1

    def num_clusters(self):
        return len(self.centroids)
