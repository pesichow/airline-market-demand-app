import requests
import pandas as pd

def fetch_flight_data():
    api_key = "89730931b2a45832248af4d847172fd8"  # Replace with your aviationstack API key
    url = f"http://api.aviationstack.com/v1/flights?access_key={api_key}&limit=100"

    response = requests.get(url)
    data = response.json()

    records = []
    for flight in data["data"]:
        records.append({
            "airline": flight["airline"]["name"],
            "from": flight["departure"]["airport"],
            "to": flight["arrival"]["airport"],
            "date": flight["flight_date"],
            "status": flight["flight_status"]
        })

    df = pd.DataFrame(records)
    df.to_csv("flight_data.csv", index=False)
    return df
