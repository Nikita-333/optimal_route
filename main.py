import math
from sklearn.cluster import KMeans
import numpy as np
from flask import Flask
from sklearn.metrics import silhouette_score
app = Flask(__name__)

class Route:
    def __init__(self, pickup_coordinates, delivery_coordinates):
        self.pickup_coordinates = pickup_coordinates
        self.delivery_coordinates = delivery_coordinates
        self.distance_matrix = self.calculate_distance_matrix()

#метод вычисления расстояния "по точкам"
    def calculate_distance(self, coord1, coord2):
        x_a, y_a = coord1
        x_b, y_b = coord2
        distance = math.sqrt(pow(x_b - x_a, 2) + pow(y_b - y_a, 2))
        return distance

#метод для вычисления матрицы расстояния
    def calculate_distance_matrix(self):
        num_pickups = len(self.pickup_coordinates)
        num_deliveries = len(self.delivery_coordinates)
        matrix = np.zeros((num_pickups, num_deliveries))

        for i in range(num_pickups):
            for j in range(num_deliveries):
                matrix[i, j] = self.calculate_distance(
                    self.pickup_coordinates[i], self.delivery_coordinates[j])
        return matrix

    def cluster_delivery_points(self, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(self.distance_matrix)
        return kmeans.labels_


@app.route("/")
@app.route("/info_route")
def info_route():
    pickup_coords = [(55.7558, 37.6173), (53.9045, 27.5615)]
    delivery_coords = [(54.6896, 25.2799), (52.5200, 13.4050)]

    route = Route(pickup_coords, delivery_coords)

    num_clusters = 2
    cluster_labels = route.cluster_delivery_points(num_clusters)

    return f"Забор: {route.pickup_coordinates}, Доставка: {route.delivery_coordinates}, " \
           f"Матрица расстояний: {route.distance_matrix}, Кластеры: {cluster_labels}"


if __name__ == "__main__":
    app.run(debug=True)
