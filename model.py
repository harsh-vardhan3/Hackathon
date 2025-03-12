import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime, timedelta
import pickle
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Define the 8 Tamil Nadu districts with their coordinates
districts = {
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "coastal": True},
    "Ramanathapuram": {"lat": 9.3639, "lon": 78.8395, "coastal": True},
    "Thoothukudi": {"lat": 8.7642, "lon": 78.1348, "coastal": True},
    "Nagapattinam": {"lat": 10.7672, "lon": 79.8449, "coastal": True},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558, "coastal": False},
    "Madurai": {"lat": 9.9252, "lon": 78.1198, "coastal": False},
    "Salem": {"lat": 11.6643, "lon": 78.1460, "coastal": False},
    "Dindigul": {"lat": 10.3624, "lon": 77.9695, "coastal": False},
}

# districts = {
#     "Jaisalmer": {"lat": 26.9124, "lon": 70.9122, "coastal": False},
#     "Bikaner": {"lat": 28.0229, "lon": 73.3119, "coastal": False},
#     "Jodhpur": {"lat": 26.2389, "lon": 73.0243, "coastal": False},
#     "Phalodi": {"lat": 27.1310, "lon": 72.3682, "coastal": False},
#     "Barmer": {"lat": 25.7463, "lon": 71.3924, "coastal": False},
# }


# Function to fetch weather data from Open-Meteo API
def fetch_weather_data(lat, lon, start_date, end_date):
    url = f"https://historical-forecast-api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_speed_80m",
            "wind_speed_120m",
            "wind_direction_10m",
            "wind_direction_80m",
            "wind_direction_120m",
            "wind_gusts_10m",
            "cloud_cover",
            "surface_pressure",
            "vapour_pressure_deficit",
        ],
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "hourly" in data:
        df = pd.DataFrame(data["hourly"])
        df["datetime"] = pd.to_datetime(df["time"])
        return df
    else:
        print(f"Error fetching data: {data}")
        return None


# Define date range for data (3 months of historical data)
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")


# Simulate collecting data for all districts
# In a real implementation, you would fetch actual data from the API
# Here we'll generate synthetic data for demonstration
def generate_synthetic_data(district_info):
    np.random.seed(42)  # For reproducibility

    # Create date range for 3 months with hourly data
    date_range = pd.date_range(start=start_date, end=end_date, freq="H")
    n_samples = len(date_range)

    # Base values with some geographical variations
    base_temp = 28 + np.random.normal(0, 3, n_samples)

    # Coastal areas have higher wind speeds
    base_wind_speed_multiplier = 1.5 if district_info["coastal"] else 1.0

    # Create dataframe with weather variables
    df = pd.DataFrame(
        {
            "datetime": date_range,
            "temperature_2m": base_temp,
            "relative_humidity_2m": 65 + np.random.normal(0, 10, n_samples),
            "wind_speed_10m": (3 + np.random.gamma(2, 1, n_samples))
            * base_wind_speed_multiplier,
            "wind_speed_80m": (5 + np.random.gamma(2.5, 1.2, n_samples))
            * base_wind_speed_multiplier,
            "wind_speed_120m": (6 + np.random.gamma(3, 1.3, n_samples))
            * base_wind_speed_multiplier,
            "wind_direction_10m": np.random.uniform(0, 360, n_samples),
            "wind_direction_80m": np.random.uniform(0, 360, n_samples),
            "wind_direction_120m": np.random.uniform(0, 360, n_samples),
            "wind_gusts_10m": (4 + np.random.gamma(3, 1.5, n_samples))
            * base_wind_speed_multiplier,
            "cloud_cover": np.random.beta(2, 5, n_samples) * 100,
            "surface_pressure": 1013 + np.random.normal(0, 3, n_samples),
            "vapour_pressure_deficit": np.random.gamma(1, 0.5, n_samples),
        }
    )
    df['latitude'] = district_info['lat']  # Add latitude column

    # Add hour of day and month features
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["day_of_year"] = df["datetime"].dt.dayofyear

    # Add some seasonal patterns
    df["temperature_2m"] += 3 * np.sin(2 * np.pi * df["day_of_year"] / 365)

    # Add diurnal patterns
    hourly_pattern = -np.cos(2 * np.pi * df["hour"] / 24)
    df["temperature_2m"] += 3 * hourly_pattern
    df["cloud_cover"] += 20 * hourly_pattern

    return df


# Create models for each energy type
class RenewableEnergyModels:
    def __init__(self):
        self.wind_model = None
        self.solar_model = None
        self.ocean_model = None
        self.wind_scaler = StandardScaler()
        self.solar_scaler = StandardScaler()
        self.ocean_scaler = StandardScaler()

    def prepare_wind_features(self, df):
        """Prepare features for wind energy prediction model"""
        # Create a copy to avoid SettingWithCopyWarning
        wind_features = df[
            [
                "wind_speed_10m",
                "wind_speed_80m",
                "wind_speed_120m",
                "wind_gusts_10m",
                "surface_pressure",
                "hour",
                "month",
            ]
        ].copy()  # Add .copy() here

        # Calculate air density
        R = 287.05
        wind_features["temperature_kelvin"] = df["temperature_2m"] + 273.15
        wind_features["air_density"] = (
            df["surface_pressure"] * 100 / (R * wind_features["temperature_kelvin"])
        )

        return wind_features

    def prepare_solar_features(self, df):
        """Prepare features for solar energy prediction model"""
        solar_features = df[
            ["temperature_2m", "cloud_cover", "hour", "month", "day_of_year"]
        ].copy()

        # Extract latitude from DataFrame (constant for all rows)
        latitude = df["latitude"].iloc[0]  # Works because all rows have the same latitude

        # Calculate solar declination (seasonal variation)
        declination = np.radians(23.45) * np.sin(2 * np.pi * (df["day_of_year"] - 81) / 365)

        # Hour angle (15° per hour from solar noon)
        hour_angle = np.radians(15 * (df["hour"] - 12))

        # Solar zenith angle formula
        solar_zenith = np.arccos(
            np.sin(np.radians(latitude)) * np.sin(declination) +
            np.cos(np.radians(latitude)) * np.cos(declination) * np.cos(hour_angle)
        )

        solar_features["solar_zenith"] = np.degrees(solar_zenith)

        # Calculate irradiance
        solar_constant = 1361  # W/m²
        solar_features["clearsky_irradiance"] = solar_constant * np.cos(solar_zenith)
        solar_features["clearsky_irradiance"] = solar_features["clearsky_irradiance"].clip(0, None)

        # Apply cloud attenuation
        solar_features["estimated_irradiance"] = solar_features["clearsky_irradiance"] * (
            1 - df["cloud_cover"] / 100 * 0.75
        )

        return solar_features

    def prepare_ocean_features(self, df):
        """Prepare features for ocean energy prediction model"""
        # Key ocean features - simplified as we don't have actual ocean data
        ocean_features = df[
            [
                "wind_speed_10m",
                "wind_gusts_10m",
                "surface_pressure",
                "wind_direction_10m",
                "hour",
                "month",
            ]
        ]

        # Calculate simplified wave height (using relationship between wind speed and wave height)
        # This is a very rough approximation
        ocean_features["estimated_wave_height"] = (
            0.0015 * df["wind_speed_10m"] ** 2 * np.log(df["wind_gusts_10m"])
        )

        # Calculate simplified tide height based on lunar cycle (extremely simplified)
        ocean_features["day_in_lunar_cycle"] = (df["day_of_year"] % 29.53).astype(int)
        ocean_features["tide_factor"] = np.sin(
            2 * np.pi * ocean_features["day_in_lunar_cycle"] / 29.53
        )

        return ocean_features

    def generate_target_values(self, features, energy_type):
        """Generate synthetic target values based on features"""
        if energy_type == "wind":
            # Wind energy is primarily related to cube of wind speed
            # In generate_target_values() for solar
            energy = (0.5 * features["air_density"] * features["wind_speed_80m"] ** 3 * 0.4)
            # Add noise
            energy = energy * (0.8 + 0.4 * np.random.random(len(energy)))
            return energy

        elif energy_type == "solar":
            # Solar energy is primarily related to irradiance and panel efficiency
            panel_efficiency = 0.18  # 18% efficient solar panel
            energy = features["estimated_irradiance"] * panel_efficiency
            # Add some noise
            energy = energy * (0.85 + 0.3 * np.random.random(len(energy)))
            return energy

        elif energy_type == "ocean":
            # Ocean energy (highly simplified)
            # Combination of wave and tidal energy
            wave_energy = 0.5 * 1025 * 9.81 * features["estimated_wave_height"] ** 2
            tidal_energy = 200 * features["tide_factor"] ** 2
            energy = wave_energy + tidal_energy
            # Add some noise
            energy = energy * (0.7 + 0.6 * np.random.random(len(energy)))
            return energy

    def train_wind_model(self, districts_data):
        """Train wind energy prediction model"""
        all_features = []
        all_targets = []

        for district, df in districts_data.items():
            features = self.prepare_wind_features(df)
            target = self.generate_target_values(features, "wind")

            all_features.append(features)
            all_targets.append(target)

        X = pd.concat(all_features)
        y = pd.concat(all_targets)

        # Check for NaN and handle
        if X.isnull().values.any() or y.isnull().values.any():
            print("Warning: NaN values detected. Imputing...")
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())

        # Scale features
        X_scaled = self.wind_scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train model
        self.wind_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.wind_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.wind_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Wind Energy Model - MSE: {mse:.2f}, R²: {r2:.2f}")

        # Feature importance
        feature_importance = pd.DataFrame(
            {"Feature": X.columns, "Importance": self.wind_model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        print("Wind Energy - Top Features:")
        print(feature_importance.head(5))

        return mse, r2, feature_importance

    def train_solar_model(self, districts_data):
        """Train solar energy prediction model"""
        all_features = []
        all_targets = []

        for district, df in districts_data.items():
            features = self.prepare_solar_features(df)
            target = self.generate_target_values(features, "solar")

            all_features.append(features)
            all_targets.append(target)

        X = pd.concat(all_features)
        y = pd.concat(all_targets)

        # Scale features
        X_scaled = self.solar_scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train model
        self.solar_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.solar_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.solar_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Solar Energy Model - MSE: {mse:.2f}, R²: {r2:.2f}")

        # Feature importance
        feature_importance = pd.DataFrame(
            {"Feature": X.columns, "Importance": self.solar_model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        print("Solar Energy - Top Features:")
        print(feature_importance.head(5))

        return mse, r2, feature_importance

    def train_ocean_model(self, districts_data):
        """Train ocean energy prediction model (only for coastal districts)"""
        all_features = []
        all_targets = []

        for district, df in districts_data.items():
            district_info = next(
                (v for k, v in districts.items() if k == district), None
            )
            if district_info and district_info["coastal"]:
                features = self.prepare_ocean_features(df)
                target = self.generate_target_values(features, "ocean")

                all_features.append(features)
                all_targets.append(target)

        if not all_features:
            print("No coastal districts data available for ocean energy model")
            return None, None, None

        X = pd.concat(all_features)
        y = pd.concat(all_targets)

        # Scale features
        X_scaled = self.ocean_scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train model
        self.ocean_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.ocean_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.ocean_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Ocean Energy Model - MSE: {mse:.2f}, R²: {r2:.2f}")

        # Feature importance
        feature_importance = pd.DataFrame(
            {"Feature": X.columns, "Importance": self.ocean_model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        print("Ocean Energy - Top Features:")
        print(feature_importance.head(5))

        return mse, r2, feature_importance

    def predict_district_potential(self, district_name, district_data):
        """Predict energy potential for a district"""
        # Prepare features
        wind_features = self.prepare_wind_features(district_data)
        solar_features = self.prepare_solar_features(district_data)

        # Scale features
        wind_features_scaled = self.wind_scaler.transform(wind_features)
        solar_features_scaled = self.solar_scaler.transform(solar_features)

        # Predict
        wind_predictions = self.wind_model.predict(wind_features_scaled)
        solar_predictions = self.solar_model.predict(solar_features_scaled)

        # For ocean energy, only predict if coastal
        district_info = next(
            (v for k, v in districts.items() if k == district_name), None
        )
        if district_info and district_info["coastal"] and self.ocean_model is not None:
            ocean_features = self.prepare_ocean_features(district_data)
            ocean_features_scaled = self.ocean_scaler.transform(ocean_features)
            ocean_predictions = self.ocean_model.predict(ocean_features_scaled)
        else:
            ocean_predictions = np.zeros(len(district_data))

        # Calculate daily averages
        district_data["wind_energy"] = wind_predictions
        district_data["solar_energy"] = solar_predictions
        district_data["ocean_energy"] = ocean_predictions

        daily_energy = district_data.groupby(district_data["datetime"].dt.date).agg(
        {"wind_energy": "sum", "solar_energy": "sum", "ocean_energy": "sum"})
        daily_energy = daily_energy / 1000  # Convert Wh to kWh

        monthly_energy = district_data.groupby(district_data["datetime"].dt.month).agg(
            {"wind_energy": "mean", "solar_energy": "mean", "ocean_energy": "mean"}
        )

        return {
            "district": district_name,
            "coastal": district_info["coastal"] if district_info else False,
            "daily_energy": daily_energy,
            "monthly_energy": monthly_energy,
            "total_potential": {
                "wind": daily_energy["wind_energy"].mean(),
                "solar": daily_energy["solar_energy"].mean(),
                "ocean": (
                    daily_energy["ocean_energy"].mean()
                    if district_info and district_info["coastal"]
                    else 0
                ),
            },
        }

    def save_models(self, directory="saved_models"):
        """Save trained models and scalers to disk"""
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save wind model and scaler
        if self.wind_model is not None:
            with open(f"{directory}/wind_model.pkl", "wb") as f:
                pickle.dump(self.wind_model, f)
            with open(f"{directory}/wind_scaler.pkl", "wb") as f:
                pickle.dump(self.wind_scaler, f)
            print(f"Wind model and scaler saved to {directory}")

        # Save solar model and scaler
        if self.solar_model is not None:
            with open(f"{directory}/solar_model.pkl", "wb") as f:
                pickle.dump(self.solar_model, f)
            with open(f"{directory}/solar_scaler.pkl", "wb") as f:
                pickle.dump(self.solar_scaler, f)
            print(f"Solar model and scaler saved to {directory}")

        # Save ocean model and scaler
        if self.ocean_model is not None:
            with open(f"{directory}/ocean_model.pkl", "wb") as f:
                pickle.dump(self.ocean_model, f)
            with open(f"{directory}/ocean_scaler.pkl", "wb") as f:
                pickle.dump(self.ocean_scaler, f)
            print(f"Ocean model and scaler saved to {directory}")

    @classmethod
    def load_models(cls, directory="saved_models"):
        """Load trained models and scalers from disk"""
        models = cls()

        # Load wind model and scaler
        try:
            with open(f"{directory}/wind_model.pkl", "rb") as f:
                models.wind_model = pickle.load(f)
            with open(f"{directory}/wind_scaler.pkl", "rb") as f:
                models.wind_scaler = pickle.load(f)
            print("Wind model and scaler loaded successfully")
        except FileNotFoundError:
            print("Wind model files not found")

        # Load solar model and scaler
        try:
            with open(f"{directory}/solar_model.pkl", "rb") as f:
                models.solar_model = pickle.load(f)
            with open(f"{directory}/solar_scaler.pkl", "rb") as f:
                models.solar_scaler = pickle.load(f)
            print("Solar model and scaler loaded successfully")
        except FileNotFoundError:
            print("Solar model files not found")

        # Load ocean model and scaler
        try:
            with open(f"{directory}/ocean_model.pkl", "rb") as f:
                models.ocean_model = pickle.load(f)
            with open(f"{directory}/ocean_scaler.pkl", "rb") as f:
                models.ocean_scaler = pickle.load(f)
            print("Ocean model and scaler loaded successfully")
        except FileNotFoundError:
            print("Ocean model files not found")

        return models


def main():
    # Fetch real weather data for each district
    districts_data = {}
    
    # Loop through districts first
    for name, info in districts.items():  # ✅ info is defined here
        print(f"Fetching data for {name}...")
        df = fetch_weather_data(info['lat'], info['lon'], start_date, end_date)

        if df is not None:
            # Add datetime features and latitude
            df['hour'] = df['datetime'].dt.hour
            df['month'] = df['datetime'].dt.month
            df['day_of_year'] = df['datetime'].dt.dayofyear
            df['latitude'] = info['lat']  # ✅ Add latitude here

            # Check for critical columns
            required_columns = ['temperature_2m', 'surface_pressure', 
                               'wind_speed_10m', 'wind_speed_80m', 'wind_speed_120m']
            
            # Skip if critical columns missing
            if not all(col in df.columns for col in required_columns):
                print(f"Skipping {name}: Missing critical columns")
                continue

            # Clean data
            df = df.dropna(subset=required_columns)
            if df.empty:
                print(f"Skipping {name}: No valid data after cleaning")
                continue

            df = df.fillna(method='ffill').fillna(method='bfill')
            districts_data[name] = df
        else:
            print(f"Failed to fetch data for {name}. Skipping.")

    # Exit if no data
    if not districts_data:
        print("No valid data for any district. Exiting.")
        return

    # Train models
    print("\nTraining renewable energy models...")
    models = RenewableEnergyModels()
    
    print("\nTraining Wind Energy Model...")
    models.train_wind_model(districts_data)
    
    print("\nTraining Solar Energy Model...")
    models.train_solar_model(districts_data)
    
    print("\nTraining Ocean Energy Model...")
    models.train_ocean_model(districts_data)

    # Save models
    print("\nSaving trained models to disk...")
    models.save_models()

    # Predict potential
    results = {}
    for name, data in districts_data.items():
        print(f"\nPredicting renewable energy potential for {name}...")
        results[name] = models.predict_district_potential(name, data)

    # Optional: Visualization

# New function to demonstrate using forecast data with saved models
def predict_with_forecast_data(forecast_data, district_name):
    """Use saved models to predict energy potential with forecast data"""
    print(f"Loading saved models for prediction...")
    models = RenewableEnergyModels.load_models()

    print(
        f"Predicting renewable energy potential for {district_name} using forecast data..."
    )
    results = models.predict_district_potential(district_name, forecast_data)

    print(f"\nRenewable Energy Forecast for {district_name}:")
    print(f"Wind Energy Potential: {results['total_potential']['wind']:.2f} kWh/m²/day")
    print(
        f"Solar Energy Potential: {results['total_potential']['solar']:.2f} kWh/m²/day"
    )

    if results["coastal"]:
        print(
            f"Ocean Energy Potential: {results['total_potential']['ocean']:.2f} kWh/m²/day"
        )

    print(
        f"Total Energy Potential: {sum(results['total_potential'].values()):.2f} kWh/m²/day"
    )

    return results


# Example of getting forecast data and using it with saved models
def get_forecast_and_predict():
    """Fetch forecast data and predict renewable energy potential"""
    # In a real implementation, you would fetch actual forecast data
    # Here's how you might structure this:

    # Define forecast date range (next 7 days)
    start_date = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

    # Example for Chennai
    district_name = "Chennai"
    district_info = districts[district_name]

    # In a real implementation, fetch actual forecast data:
    forecast_data = fetch_weather_data(district_info['lat'], district_info['lon'], start_date, end_date)

    # For demonstration, we'll generate synthetic forecast data
    # print(f"Generating synthetic forecast data for {district_name}...")
    # forecast_data = fetch_weather_data(district_info)

    # Make predictions using saved models
    results = predict_with_forecast_data(forecast_data, district_name)

    # Visualize forecast
    plt.figure(figsize=(12, 6))

    # Plot daily forecast
    daily_energy = results["daily_energy"]
    plt.plot(
        daily_energy.index[-7:], daily_energy["wind_energy"][-7:], label="Wind Energy"
    )
    plt.plot(
        daily_energy.index[-7:], daily_energy["solar_energy"][-7:], label="Solar Energy"
    )

    if results["coastal"]:
        plt.plot(
            daily_energy.index[-7:],
            daily_energy["ocean_energy"][-7:],
            label="Ocean Energy",
        )

    plt.xlabel("Date")
    plt.ylabel("Energy Potential (kWh/m²/day)")
    plt.title(f"{district_name} 7-Day Renewable Energy Forecast")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"{district_name.lower()}_renewable_forecast.png")
    plt.show()


if __name__ == "__main__":
    # First run: train and save models
    main()

    # Second run: use saved models with forecast data
    # Uncomment this line to run prediction with forecast data
    # get_forecast_and_predict()
