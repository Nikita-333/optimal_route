import math
import numpy as np
import folium
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import PredefinedSplit, ParameterGrid
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from data import limited_pickup_coords, limited_dropoff_coords


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

    def calculate_distance(self, coord1, coord2, units: str = 'deg'):
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
        d = math.sqrt(pow(x_b - x_a, 2) + pow(y_b - y_a, 2))

        if units == 'km':
            # 111 kilometers per 1 degree
            r = 111
            # Calculate the distance in km
            distance = d * r
        else:
            distance = d

        return distance

    def visualize_clusters(self, eps, min_samples, distance_units='deg'):

        cluster_labels = self.cluster_delivery_points(eps, min_samples)

        center_latitude = sum(lat for lat, lon in self.delivery_coordinates) / len(self.delivery_coordinates)
        center_longitude = sum(lon for lat, lon in self.delivery_coordinates) / len(self.delivery_coordinates)
        m = folium.Map(location=[center_latitude, center_longitude], zoom_start=10)

        for coord, cluster_label in zip(self.delivery_coordinates, cluster_labels):
            if cluster_label != -1:
                color = 'black' if cluster_label == 0 else 'blue' if cluster_label == 1 else 'purple'
                folium.Marker(location=[coord[0], coord[1]], popup=f'Cluster {cluster_label}',
                              icon=folium.Icon(color=color)).add_to(m)

        # Save the map to an HTML file
        m.save('clustered_map.html')

    def calculate_distance_matrix(self, distance_units='deg'):
        num_deliveries = len(self.delivery_coordinates)
        matrix = np.zeros((num_deliveries, num_deliveries))

        for i in range(num_deliveries):
            for j in range(num_deliveries):
                matrix[i, j] = self.calculate_distance(
                    self.delivery_coordinates[i], self.delivery_coordinates[j], distance_units)
        return matrix

    def cluster_delivery_points(self, eps, min_samples, distance_units='deg'):
        """
        Calculate DBSCAN clustering.
            :param eps: The maximum distance between samples for
                        them to be considered as in the same neighborhood.
            :param min_samples: The number of samples in a neighborhood
                                for a point to be considered as a core point.
            :param distance_units: Distance units for eps and min_samples: 'km' or 'deg'.
            :return: Array of cluster labels for each delivery point.
        """
        if distance_units == 'deg':
            self.distance_matrix = self.calculate_distance_matrix(distance_units='km')
            eps_km = eps * 111
            min_samples_km = min_samples * 111
        elif distance_units == 'km':
            self.distance_matrix = self.calculate_distance_matrix(distance_units='deg')
            conversion_factor = 1 / 111  # Convert km to degrees
            eps_km = eps
            min_samples_km = min_samples
        else:
            raise ValueError("Invalid distance_units. Use 'km' or 'deg'.")

        dbscan = DBSCAN(eps=eps_km, min_samples=min_samples_km, metric='precomputed')
        dbscan.fit(self.distance_matrix)
        return dbscan.labels_

    def assign_clusters_to_routes(self, eps, min_samples, distance_units='deg'):
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

    def find_optimal_parameters(self, distance_units='deg'):
        best_score = -1
        best_eps = 0
        best_min_samples = 0

        for eps in range(1, 500):
            for min_samples in range(1, 10):
                cluster_labels = self.cluster_delivery_points(eps, min_samples, distance_units)
                try:
                    silhouette_avg = silhouette_score(self.distance_matrix, cluster_labels)
                except:
                    silhouette_avg = -1

                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_eps = eps
                    best_min_samples = min_samples

        return best_eps, best_min_samples

route = Route(limited_pickup_coords, limited_dropoff_coords)
eps = 1
min_samples = 4

#best_eps, best_min_samples = route.find_optimal_parameters("km")

#print(f"Оптимальные параметры - eps: {best_eps}, min_samples: {best_min_samples}")
#route.visualize_clusters(eps_km, min_samples, "km")

print(route.assign_clusters_to_routes(eps, min_samples, "km"))
