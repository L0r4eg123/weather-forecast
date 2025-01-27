import requests
import pandas as pd
import datetime
import tkinter as tk
from tkinter import messagebox
from tkinter import Canvas
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def train_and_save_model():
    data = {
        "RelativeHumidity": [50, 60, 70, 80, 90],
        "DewPoint": [10, 12, 14, 16, 18],
        "ApparentTemperature": [5, 6, 7, 8, 9],
        "PrecipitationProbability": [10, 20, 30, 40, 50],
        "Precipitation": [0, 0.2, 0.5, 1.0, 2.0],
        "Rain": [0, 0.1, 0.3, 0.7, 1.5],
        "Showers": [0, 0.1, 0.2, 0.5, 1.0],
        "Snowfall": [0, 0, 0.1, 0.2, 0.5],
        "SnowDepth": [0, 0, 1, 2, 5],
        "WindSpeed10m": [5, 10, 15, 20, 25],
        "WindSpeed80m": [7, 12, 17, 22, 27],
        "WindSpeed120m": [8, 13, 18, 23, 28],
        "WindSpeed180m": [9, 14, 19, 24, 29],
        "WindDirection10m": [90, 100, 110, 120, 130],
        "WindDirection80m": [95, 105, 115, 125, 135],
        "WindDirection120m": [100, 110, 120, 130, 140],
        "WindDirection180m": [105, 115, 125, 135, 145],
        "WindGusts10m": [15, 20, 25, 30, 35],
        "Temperature80m": [4, 5, 6, 7, 8],
        "Temperature120m": [3, 4, 5, 6, 7],
        "Temperature180m": [2, 3, 4, 5, 6],
        "Month": [1, 1, 1, 1, 1],
        "TargetTemperature": [0, 2, 4, 6, 8]
    }
    weather_data = pd.DataFrame(data)

    
    X = weather_data.drop(columns=["TargetTemperature"])
    y = weather_data["TargetTemperature"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = GradientBoostingRegressor()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)


    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    
    joblib.dump(best_model, "weather_forecast_model.pkl")

def ai_weather_forecast(features):
    model = joblib.load("weather_forecast_model.pkl")
    input_data = pd.DataFrame([features])
    predicted_temperature = model.predict(input_data)[0]
    return predicted_temperature


def get_weather_forecast(city_name, target_date):
    geocode_url = f"https://nominatim.openstreetmap.org/search?city={city_name}&format=json"
    headers = {"User-Agent": "WeatherForecastApp/1.0 (contact@example.com)"}
    response = requests.get(geocode_url, headers=headers)
    if not response.ok:
        raise ValueError("Kļūda, pieslēdzoties ģeokodēšanas API. Pārbaudiet interneta savienojumu.")
    response_data = response.json()
    if not response_data:
        raise ValueError(f"Neizdevās atrast pilsētu '{city_name}'. Pārbaudiet pareizrakstību latīņu burtiem.")

    location = response_data[0]
    lat, lon = float(location['lat']), float(location['lon'])

    weather_url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=relative_humidity_2m,dewpoint_2m,apparent_temperature,precipitation_probability,precipitation,rain,showers,snowfall,snow_depth,windspeed_10m,windspeed_80m,windspeed_120m,windspeed_180m,winddirection_10m,winddirection_80m,winddirection_120m,winddirection_180m,windgusts_10m,temperature_80m,temperature_120m,temperature_180m"
                   f"&timezone=auto")
    weather_response = requests.get(weather_url, headers=headers)
    if not weather_response.ok:
        raise ValueError("Neizdevās iegūt laikapstākļu datus. Pārbaudiet piekļuvi Open-Meteo API.")

    weather_data = weather_response.json()
    if 'hourly' not in weather_data:
        raise ValueError("Saņemti nekorekti vai nepilnīgi laikapstākļu dati no Open-Meteo.")

    target_datetime = datetime.datetime.strptime(target_date, "%d.%m.%Y").date()

    hourly_data = pd.DataFrame(weather_data['hourly'])
    hourly_data['time'] = pd.to_datetime(hourly_data['time'])
    hourly_data = hourly_data[hourly_data['time'].dt.date == target_datetime]

    if hourly_data.empty:
        return f"Prognoze izvēlētajam datumam ({target_date}) nav pieejama. Pamēģiniet citu datumu."

    result = f"Laikapstākļu prognoze pilsētā {city_name} ({target_date}):\n"
    for _, row in hourly_data.iterrows():
        features = {
            "RelativeHumidity": row.get("relative_humidity_2m", 0),
            "DewPoint": row.get("dewpoint_2m", 0),
            "ApparentTemperature": row.get("apparent_temperature", 0),
            "PrecipitationProbability": row.get("precipitation_probability", 0),
            "Precipitation": row.get("precipitation", 0),
            "Rain": row.get("rain", 0),
            "Showers": row.get("showers", 0),
            "Snowfall": row.get("snowfall", 0),
            "SnowDepth": row.get("snow_depth", 0),
            "WindSpeed10m": row.get("windspeed_10m", 0),
            "WindSpeed80m": row.get("windspeed_80m", 0),
            "WindSpeed120m": row.get("windspeed_120m", 0),
            "WindSpeed180m": row.get("windspeed_180m", 0),
            "WindDirection10m": row.get("winddirection_10m", 0),
            "WindDirection80m": row.get("winddirection_80m", 0),
            "WindDirection120m": row.get("winddirection_120m", 0),
            "WindDirection180m": row.get("winddirection_180m", 0),
            "WindGusts10m": row.get("windgusts_10m", 0),
            "Temperature80m": row.get("temperature_80m", 0),
            "Temperature120m": row.get("temperature_120m", 0),
            "Temperature180m": row.get("temperature_180m", 0),
            "Month": target_datetime.month
        }

        predicted_temperature = ai_weather_forecast(features)
        time = row['time'].strftime("%H:%M")
        result += f"{time} - Prognozētā temperatūra: {predicted_temperature:.2f}°C, Nokrišņi: {features['Precipitation']} mm\n"
    return result


def display_forecast():
    city_name = city_entry.get()
    target_date = date_entry.get()

    try:
        result = get_weather_forecast(city_name, target_date)
        messagebox.showinfo("Prognoze", result)
    except Exception as e:
        messagebox.showerror("Kļūda", str(e))

def draw_clouds(canvas):
    canvas.create_oval(50, 50, 150, 100, fill="white", outline="white")
    canvas.create_oval(100, 40, 200, 90, fill="white", outline="white")
    canvas.create_oval(150, 50, 250, 100, fill="white", outline="white")
    canvas.create_oval(300, 100, 400, 150, fill="white", outline="white")
    canvas.create_oval(350, 90, 450, 140, fill="white", outline="white")
    canvas.create_oval(400, 100, 500, 150, fill="white", outline="white")

def main():
    global city_entry, date_entry

  
    train_and_save_model()

    root = tk.Tk()
    root.title("Laikapstākļu prognoze")
    root.geometry("500x500")
    root.configure(bg="#87CEEB")

    canvas = tk.Canvas(root, width=500, height=200, bg="#87CEEB", highlightthickness=0)
    canvas.pack()
    draw_clouds(canvas)

    font_large = ("Arial", 16, "bold")
    font_medium = ("Arial", 14)

    tk.Label(root, text="Pilsētas nosaukums (latīņu burtiem):", font=font_large, bg="#87CEEB").pack(pady=10)
    city_entry = tk.Entry(root, width=40, font=font_medium)
    city_entry.pack(pady=10)

    tk.Label(root, text="Datums prognozei (formātā DD.MM.GGGG):", font=font_large, bg="#87CEEB").pack(pady=10)
    date_entry = tk.Entry(root, width=40, font=font_medium)
    date_entry.pack(pady=10)

    tk.Button(root, text="Saņemt prognozi", font=font_large, command=display_forecast).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()