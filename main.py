import math
from sklearn.cluster import KMeans
import numpy as np
from flask import Flask
from sklearn.metrics import silhouette_score

from Route import Route

app = Flask(__name__)
"""
The main part of the script that creates a Flask 
web application and defines a route to display 
information about a route.

The Route class is used to create a route 
instance and perform clustering on delivery points.
"""
@app.route("/")
@app.route("/info_route")
def info_route():
    # List of tuples containing pickup coordinates.
    pickup_coords = [(55.7558, 37.6173), (53.9045, 27.5615), (50.4501, 30.5234),
                     (59.3293, 18.0686), (48.8566, 2.3522), (41.9028, 12.4964),
                     (52.3792, 4.8994), (40.7128, -74.0060), (37.7749, -122.4194),
                     (34.0522, -118.2437), (35.6895, 139.6917), (31.9686, -99.9018),
                     (51.1657, 10.4515), (52.384742, 30.350130)]
    # List of tuples containing delivery coords
    delivery_coords = [(54.6896, 25.2799), (52.5200, 13.4050), (48.8566, 2.3522),
                       (55.7558, 37.6173), (37.7749, -122.4194), (45.4642, 9.1900),
                       (51.5074, -0.1278), (34.0522, -118.2437), (40.7128, -74.0060),
                       (41.9028, 12.4964), (35.6895, 139.6917), (29.7604, -95.3698),
                       (52.5200, 13.4050), (52.383002, 30.364431)]

    route = Route(pickup_coords, delivery_coords)

    eps = 0.5
    min_samples = 2

    cluster_labels = route.cluster_delivery_points(eps, min_samples)

    return f"Забор: {route.pickup_coordinates}, Доставка: {route.delivery_coordinates}, " \
           f"Матрица расстояний: {route.distance_matrix}, Кластеры: {cluster_labels}"


if __name__ == "__main__":
    app.run(debug=True)
