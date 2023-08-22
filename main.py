import math
from sklearn.cluster import KMeans
import numpy as np
from flask import Flask
from sklearn.metrics import silhouette_score

from Route import Route

app = Flask(__name__)

@app.route("/")
@app.route("/info_route")
def info_route():
    pickup_coords = [(55.7558, 37.6173), (53.9045, 27.5615)]
    delivery_coords = [(54.6896, 25.2799), (52.5200, 13.4050)]

    route = Route(pickup_coords, delivery_coords)

    eps = 0.5
    min_samples = 2

    cluster_labels = route.cluster_delivery_points(eps, min_samples)

    return f"Забор: {route.pickup_coordinates}, Доставка: {route.delivery_coordinates}, " \
           f"Матрица расстояний: {route.distance_matrix}, Кластеры: {cluster_labels}"


if __name__ == "__main__":
    app.run(debug=True)
