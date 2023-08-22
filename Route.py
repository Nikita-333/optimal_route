import math
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class Route:
    def __init__(self, pickup_coordinates, delivery_coordinates):
        """
        Initialize a Route instance with pickup and delivery coordinates.
        :param:
            List of tuples containing pickup_coordinates.
            List of tuples containing delivery_coordinates.
        Matrix of distances between delivery coordinates.
        """
        self.pickup_coordinates = pickup_coordinates
        self.delivery_coordinates = delivery_coordinates
        self.distance_matrix = self.calculate_distance_matrix()

    def calculate_distance(self, coord1, coord2):
        """
        Calculate the Euclidean distance between two coordinates.
        :param:
            coord1:The first coordinate (x, y).
            coord2: The second coordinate (x, y).
        Returns:
            float: The Euclidean distance between coord1 and coord2.
        """
        x_a, y_a = coord1
        x_b, y_b = coord2
        distance = math.sqrt(pow(x_b - x_a, 2) + pow(y_b - y_a, 2))
        return distance

    def calculate_distance_matrix(self):
        """
        Calculate the distance matrix for all delivery coordinates.
            Returns:
                A matrix of distances between all pairs of delivery coordinates.
        """
        num_deliveries = len(self.delivery_coordinates)
        matrix = np.zeros((num_deliveries, num_deliveries))

        for i in range(num_deliveries):
            for j in range(num_deliveries):
                matrix[i, j] = self.calculate_distance(
                    self.delivery_coordinates[i], self.delivery_coordinates[j])
        return matrix

    def cluster_delivery_points(self, eps, min_samples):
        """
        Calculate DBSCAN clustering.
            :param:
                eps: The maximum distance between samples for
                them to be considered as in the same neighborhood.

                min_samples: The number of samples in a neighborhood
                for a point to be considered as a core point.

            Returns:
                Array of cluster labels for each delivery point.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        dbscan.fit(self.distance_matrix)
        return dbscan.labels_

    def visualize_clusters(self, eps, min_samples):
        """
        Visualize the clustered delivery points using a scatter plot.
            :param:
                eps: The maximum distance between samples for DBSCAN clustering.
                min_samples: The minimum number of samples for DBSCAN clustering.
        """
        cluster_labels = self.cluster_delivery_points(eps, min_samples)

        x_coords = [coord[0] for coord in self.delivery_coordinates]
        y_coords = [coord[1] for coord in self.delivery_coordinates]

        plt.figure(figsize=(5, 5))
        plt.scatter(x_coords, y_coords, c=cluster_labels, cmap='rainbow', s=60)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Clustered Delivery Points')
        plt.show()

    def assign_clusters_to_routes(self, eps, min_samples):
        """
        Assign cluster labels to the routes based on DBSCAN clustering.
            :param:
                eps: The maximum distance between samples for DBSCAN clustering.
                min_samples: The minimum number of samples for DBSCAN clustering.

            Returns:
                dict: A dictionary where keys are cluster labels and values are lists of route indices.
                """
        cluster_labels = self.cluster_delivery_points(eps, min_samples)
        clusters_to_routes = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters_to_routes:
                clusters_to_routes[label] = []
            clusters_to_routes[label].append(i)
        return clusters_to_routes

#List of tuples containing pickup coordinates.
pickup_coords = [(55.7558, 37.6173), (53.9045, 27.5615), (50.4501, 30.5234),
                 (59.3293, 18.0686), (48.8566, 2.3522), (41.9028, 12.4964),
                 (52.3792, 4.8994), (40.7128, -74.0060), (37.7749, -122.4194),
                 (34.0522, -118.2437), (35.6895, 139.6917), (31.9686, -99.9018),
                 (51.1657, 10.4515), (52.384742, 30.350130)]
#List of tuples containing delivery coords
delivery_coords = [(54.6896, 25.2799), (52.5200, 13.4050), (48.8566, 2.3522),
                   (55.7558, 37.6173), (37.7749, -122.4194), (45.4642, 9.1900),
                   (51.5074, -0.1278), (34.0522, -118.2437), (40.7128, -74.0060),
                   (41.9028, 12.4964), (35.6895, 139.6917), (29.7604, -95.3698),
                   (52.5200, 13.4050), (52.383002, 30.364431)]

route = Route(pickup_coords, delivery_coords)

