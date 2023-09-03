import pandas as pd
import folium
from sklearn.cluster import DBSCAN

data = pd.read_csv("nyc_taxi_data_2014.csv",low_memory=False)

df = pd.DataFrame(data)

df.drop(columns=['vendor_id', 'pickup_datetime', 'dropoff_datetime', 'passenger_count', 'trip_distance', 'rate_code',
                 'store_and_fwd_flag', 'payment_type', 'fare_amount', 'surcharge', 'mta_tax', 'tip_amount',
                 'tolls_amount', 'total_amount'], inplace=True)

df.dropna(subset=['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'], inplace=True)

pickup_coords = [(lat, lon) for lat, lon in zip(df['pickup_latitude'], df['pickup_longitude'])]
dropoff_coords = [(lat, lon) for lat, lon in zip(df['dropoff_latitude'], df['dropoff_longitude'])]


def visualize_coords(coords_pickup, coords_dropoff):
    center_latitude = (
                sum(lat for lat, lon in coords_pickup + coords_dropoff) / (len(coords_pickup) + len(coords_dropoff)))
    center_longitude = (
                sum(lon for lat, lon in coords_pickup + coords_dropoff) / (len(coords_pickup) + len(coords_dropoff)))
    initial_location = (center_latitude, center_longitude)

    m = folium.Map(location=initial_location, zoom_start=10)

    for coord in coords_pickup:
        folium.Marker(location=coord, popup='Pickup', icon=folium.Icon(color='red')).add_to(m)

    for coord in coords_dropoff:
        folium.Marker(location=coord, popup='Dropoff', icon=folium.Icon(color='green')).add_to(m)

    m.save('map.html')


limited_pickup_coords = pickup_coords[:500]
limited_dropoff_coords = dropoff_coords[:500]
