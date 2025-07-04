import pandas as pd

def process_flight_data(csv_file="flight_data.csv"):
    df = pd.read_csv(csv_file)
    df['route'] = df['from'] + " → " + df['to']
    popular_routes = df['route'].value_counts().reset_index()
    popular_routes.columns = ['route', 'count']
    return popular_routes
