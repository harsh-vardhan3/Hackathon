import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import requests
import json
from datetime import datetime, timedelta
import os
import pickle
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Function to fetch forecast data from Open-Meteo API
def fetch_forecast_data(lat, lon, forecast_days=7):
    """
    Fetch weather forecast data for renewable energy prediction
    
    Parameters:
    lat (float): Latitude of the location
    lon (float): Longitude of the location
    forecast_days (int): Number of days to forecast (max 16 days)
    
    Returns:
    pandas.DataFrame: Hourly forecast data
    """
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': [
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
        'forecast_days': forecast_days,
        'timezone': 'auto'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        if 'hourly' in data:
            df = pd.DataFrame(data['hourly'])
            df['datetime'] = pd.to_datetime(df['time'])
            return df
        else:
            print(f"Error in response format: {data}")
            return None
    else:
        print(f"Error fetching forecast data: {response.status_code}")
        return None

class RenewableEnergyModels:
    def __init__(self, model_dir='./saved_models'):
        """
        Load pre-trained models and scalers from pickle files
        
        Parameters:
        model_dir (str): Directory containing the saved model files
        """
        self.model_dir = model_dir
        
        # Load wind models
        self.wind_model = self._load_pickle(os.path.join(model_dir, 'wind_model.pkl'))
        self.wind_scaler = self._load_pickle(os.path.join(model_dir, 'wind_scaler.pkl'))
        
        # Load solar models
        self.solar_model = self._load_pickle(os.path.join(model_dir, 'solar_model.pkl'))
        self.solar_scaler = self._load_pickle(os.path.join(model_dir, 'solar_scaler.pkl'))
        
        # Load ocean models if available
        ocean_model_path = os.path.join(model_dir, 'ocean_model.pkl')
        if os.path.exists(ocean_model_path):
            self.ocean_model = self._load_pickle(ocean_model_path)
            self.ocean_scaler = self._load_pickle(os.path.join(model_dir, 'ocean_scaler.pkl'))
        else:
            self.ocean_model = None
            self.ocean_scaler = None
    
    def _load_pickle(self, file_path):
        """Load a pickle file"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def prepare_wind_features(self, data):
        """
        Prepare features for wind energy prediction
        
        Parameters:
        data (DataFrame): Weather data
        
        Returns:
        DataFrame: Features for wind energy prediction
        """
        features = data[[
                "wind_speed_10m",
                "wind_speed_80m",
                "wind_speed_120m",
                "wind_gusts_10m",
                "surface_pressure",
                "hour",
                "month",
        ]].copy()
        
        # Convert wind direction to sine and cosine components
        R = 287.05  # Gas constant for dry air in J/(kg·K)
        features["temperature_kelvin"] = data["temperature_2m"] + 273.15
        features["air_density"] = (
            data["surface_pressure"] * 100 / (R * features["temperature_kelvin"])
        )

        return features
    
    def prepare_solar_features(self, data):
        """
        Prepare features for solar energy prediction
        
        Parameters:
        data (DataFrame): Weather data
        
        Returns:
        DataFrame: Features for solar energy prediction
        """
        features = data[[
            "temperature_2m", "cloud_cover", "hour", "month", "day_of_year"
        ]].copy()
        
        features["solar_zenith"] = 90 - 70 * np.sin(
            np.pi * (data["hour"] - 6) / 12
        ) * np.sin(np.pi * data["day_of_year"] / 365)

        solar_constant = 1361

        features["clearsky_irradiance"] = solar_constant * np.cos(
            np.radians(features["solar_zenith"].clip(0, 90))
        )
        features["clearsky_irradiance"] = features[
            "clearsky_irradiance"
        ].clip(0, None)

        features["estimated_irradiance"] = features[
            "clearsky_irradiance"
        ] * (1 - data["cloud_cover"] / 100 * 0.75)

        return features
    
    def prepare_ocean_features(self, data):
        """
        Prepare features for ocean energy prediction
        
        Parameters:
        data (DataFrame): Weather data
        
        Returns:
        DataFrame: Features for ocean energy prediction
        """
        features = data[[
                "wind_speed_10m",
                "wind_gusts_10m",
                "surface_pressure",
                "wind_direction_10m",
                "hour",
                "month",
        ]].copy()
        
        features["estimated_wave_height"] = (
            0.0015 * data["wind_speed_10m"] ** 2 * np.log(data["wind_gusts_10m"])
        )

        # Calculate simplified tide height based on lunar cycle (extremely simplified)
        features["day_in_lunar_cycle"] = (data["day_of_year"] % 29.53).astype(int)
        features["tide_factor"] = np.sin(
            2 * np.pi * features["day_in_lunar_cycle"] / 29.53
        )

        return features

class RenewableEnergyForecaster:
    def __init__(self, trained_models):
        """
        Initialize forecaster with pre-trained models
        
        Parameters:
        trained_models (RenewableEnergyModels): Pre-trained models for wind, solar, and ocean energy
        """
        self.models = trained_models
    
    def forecast_district_energy(self, district_name, district_info, forecast_days=7):
        """
        Generate energy potential forecast for a specific district
        
        Parameters:
        district_name (str): Name of the district
        district_info (dict): Information about the district including lat, lon, coastal status
        forecast_days (int): Number of days to forecast
        
        Returns:
        dict: Forecast results including daily and hourly predictions
        """
        # Fetch forecast data
        print(f"Fetching forecast data for {district_name}...")
        forecast_data = fetch_forecast_data(
            district_info['lat'], 
            district_info['lon'], 
            forecast_days
        )
        
        if forecast_data is None:
            return None
        
        # Add hour and month features
        forecast_data['hour'] = forecast_data['datetime'].dt.hour
        forecast_data['month'] = forecast_data['datetime'].dt.month
        forecast_data['day_of_year'] = forecast_data['datetime'].dt.dayofyear
        
        # Prepare features for prediction
        wind_features = self.models.prepare_wind_features(forecast_data)
        solar_features = self.models.prepare_solar_features(forecast_data)
        
        # Scale features
        wind_features_scaled = self.models.wind_scaler.transform(wind_features)
        solar_features_scaled = self.models.solar_scaler.transform(solar_features)
        
        # Make predictions
        wind_predictions = self.models.wind_model.predict(wind_features_scaled)
        solar_predictions = self.models.solar_model.predict(solar_features_scaled)
        
        # For ocean energy, only predict if coastal
        if district_info['coastal'] and self.models.ocean_model is not None:
            ocean_features = self.models.prepare_ocean_features(forecast_data)
            ocean_features_scaled = self.models.ocean_scaler.transform(ocean_features)
            ocean_predictions = self.models.ocean_model.predict(ocean_features_scaled)
        else:
            ocean_predictions = np.zeros(len(forecast_data))
        
        # Add predictions to the dataframe
        forecast_data['wind_energy'] = wind_predictions
        forecast_data['solar_energy'] = solar_predictions
        forecast_data['ocean_energy'] = ocean_predictions
        forecast_data['total_energy'] = wind_predictions + solar_predictions + ocean_predictions
        
        # Aggregate by day
        daily_forecast = forecast_data.groupby(forecast_data['datetime'].dt.date).agg({
            'wind_energy': 'sum',
            'solar_energy': 'sum',
            'ocean_energy': 'sum',
            'total_energy': 'sum'
        })
        
        # Calculate hourly average patterns
        hourly_patterns = forecast_data.groupby(forecast_data['datetime'].dt.hour).agg({
            'wind_energy': 'mean',
            'solar_energy': 'mean',
            'ocean_energy': 'mean',
            'total_energy': 'mean'
        })
        
        return {
            'district': district_name,
            'forecast_period': {
                'start': forecast_data['datetime'].min().strftime('%Y-%m-%d'),
                'end': forecast_data['datetime'].max().strftime('%Y-%m-%d')
            },
            'hourly_forecast': forecast_data[[
                'datetime', 'wind_energy', 'solar_energy', 
                'ocean_energy', 'total_energy'
            ]],
            'daily_forecast': daily_forecast,
            'hourly_patterns': hourly_patterns,
            'total_potential': {
                'wind': daily_forecast['wind_energy'].mean(),
                'solar': daily_forecast['solar_energy'].mean(),
                'ocean': daily_forecast['ocean_energy'].mean() if district_info['coastal'] else 0,
                'total': daily_forecast['total_energy'].mean()
            }
        }

def generate_forecast_report(district_forecasts):
    """
    Generate a comprehensive forecast report for all districts
    
    Parameters:
    district_forecasts (dict): Dictionary with district names as keys and forecast results as values
    
    Returns:
    dict: Summary report of renewable energy forecasts
    """
    # Calculate total potential for the entire state
    total_wind = sum(d['total_potential']['wind'] for d in district_forecasts.values())
    total_solar = sum(d['total_potential']['solar'] for d in district_forecasts.values())
    total_ocean = sum(d['total_potential']['ocean'] for d in district_forecasts.values())
    total_energy = total_wind + total_solar + total_ocean
    
    # Find best districts for each energy type
    best_wind_district = max(district_forecasts.items(), key=lambda x: x[1]['total_potential']['wind'])[0]
    best_solar_district = max(district_forecasts.items(), key=lambda x: x[1]['total_potential']['solar'])[0]
    best_ocean_district = max(district_forecasts.items(), key=lambda x: x[1]['total_potential']['ocean'])[0]
    
    # Format the report
    report = {
        'forecast_period': next(iter(district_forecasts.values()))['forecast_period'],
        'state_summary': {
            'total_energy_potential': total_energy,
            'energy_mix': {
                'wind': (total_wind / total_energy) * 100 if total_energy > 0 else 0,
                'solar': (total_solar / total_energy) * 100 if total_energy > 0 else 0,
                'ocean': (total_ocean / total_energy) * 100 if total_energy > 0 else 0
            }
        },
        'best_districts': {
            'wind': best_wind_district,
            'solar': best_solar_district,
            'ocean': best_ocean_district
        },
        'district_rankings': {
            'total_energy': sorted(
                [(d, f['total_potential']['total']) for d, f in district_forecasts.items()],
                key=lambda x: x[1], reverse=True
            ),
            'wind_energy': sorted(
                [(d, f['total_potential']['wind']) for d, f in district_forecasts.items()],
                key=lambda x: x[1], reverse=True
            ),
            'solar_energy': sorted(
                [(d, f['total_potential']['solar']) for d, f in district_forecasts.items()],
                key=lambda x: x[1], reverse=True
            ),
            'ocean_energy': sorted(
                [(d, f['total_potential']['ocean']) for d, f in district_forecasts.items()],
                key=lambda x: x[1], reverse=True
            )
        }
    }
    
    return report

def visualize_forecasts(district_forecasts):
    """
    Create visualizations for energy forecasts
    
    Parameters:
    district_forecasts (dict): Dictionary with district forecasts
    """
    # 1. Daily forecast for one district (e.g., Chennai)
    if 'Chennai' in district_forecasts:
        chennai_forecast = district_forecasts['Chennai']
        
        plt.figure(figsize=(12, 6))
        
        # Get daily forecast
        daily = chennai_forecast['daily_forecast']
        dates = daily.index
        
        plt.plot(dates, daily['wind_energy'], label='Wind Energy', marker='o')
        plt.plot(dates, daily['solar_energy'], label='Solar Energy', marker='s')
        
        if chennai_forecast['total_potential']['ocean'] > 0:
            plt.plot(dates, daily['ocean_energy'], label='Ocean Energy', marker='^')
            
        plt.plot(dates, daily['total_energy'], label='Total Energy', 
                 linestyle='--', color='black', linewidth=2)
        
        plt.title('7-Day Renewable Energy Forecast for Chennai')
        plt.xlabel('Date')
        plt.ylabel('Energy Potential (kWh/m²)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig('chennai_energy_forecast.png')
        plt.close()
    
    # 2. Compare all districts' total potential
    plt.figure(figsize=(12, 6))
    
    districts = list(district_forecasts.keys())
    wind_values = [df['total_potential']['wind'] for df in district_forecasts.values()]
    solar_values = [df['total_potential']['solar'] for df in district_forecasts.values()]
    ocean_values = [df['total_potential']['ocean'] for df in district_forecasts.values()]
    
    x = np.arange(len(districts))
    width = 0.25
    
    plt.bar(x - width, wind_values, width, label='Wind', color='skyblue')
    plt.bar(x, solar_values, width, label='Solar', color='orange')
    plt.bar(x + width, ocean_values, width, label='Ocean', color='teal')
    
    plt.xlabel('District')
    plt.ylabel('Average Daily Energy Potential (kWh/m²)')
    plt.title('Forecasted Renewable Energy Potential by District')
    plt.xticks(x, districts, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('district_energy_comparison.png')
    plt.close()
    
    # 3. Hourly pattern visualization for Chennai
    if 'Chennai' in district_forecasts:
        hourly_pattern = district_forecasts['Chennai']['hourly_patterns']
        
        plt.figure(figsize=(12, 6))
        
        hours = hourly_pattern.index
        
        plt.plot(hours, hourly_pattern['wind_energy'], label='Wind Energy')
        plt.plot(hours, hourly_pattern['solar_energy'], label='Solar Energy')
        
        if district_forecasts['Chennai']['total_potential']['ocean'] > 0:
            plt.plot(hours, hourly_pattern['ocean_energy'], label='Ocean Energy')
            
        plt.plot(hours, hourly_pattern['total_energy'], label='Total Energy', 
                 linestyle='--', color='black', linewidth=2)
        
        plt.title('Average Hourly Energy Generation Pattern (Chennai)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Energy Potential (kWh/m²)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(range(0, 24, 2))
        plt.tight_layout()
        
        plt.savefig('chennai_hourly_pattern.png')
        plt.close()

# Define districts with their coordinates and coastal status
districts = {
    'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'coastal': True},
    'Coimbatore': {'lat': 11.0168, 'lon': 76.9558, 'coastal': False},
    'Madurai': {'lat': 9.9252, 'lon': 78.1198, 'coastal': False},
    'Trichy': {'lat': 10.7905, 'lon': 78.7047, 'coastal': False},
    'Salem': {'lat': 11.6643, 'lon': 78.1460, 'coastal': False},
    'Vellore': {'lat': 12.9165, 'lon': 79.1325, 'coastal': False},
    'Kanyakumari': {'lat': 8.0883, 'lon': 77.5385, 'coastal': True},
    'Tuticorin': {'lat': 8.7642, 'lon': 78.1348, 'coastal': True}
}

# districts = {
#     # Arid desert regions with some of India's highest solar potential
#     "Jaisalmer": {"lat": 26.9124, "lon": 70.9122, "coastal": False},  # Thar Desert core
#     "Bikaner": {"lat": 28.0229, "lon": 73.3119, "coastal": False},     # High solar parks
#     "Jodhpur": {"lat": 26.2389, "lon": 73.0243, "coastal": False},     # Sunny & dry climate
#     "Phalodi": {"lat": 27.1310, "lon": 72.3682, "coastal": False},     # Hottest place in India
#     "Barmer": {"lat": 25.7463, "lon": 71.3924, "coastal": False},      # Arid with vast open spaces
# }

# Example usage
def main():
    # Check if model directory exists
    model_dir = '/Users/sanchitbishnoi/Desktop/Ahouba/Hackathon/saved_models'
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found.")
        return
    
    # Initialize forecaster with pre-trained models
    try:
        models = RenewableEnergyModels(model_dir)
        
        # Check if models loaded successfully
        if models.wind_model is None or models.solar_model is None:
            print("Error: Failed to load required models.")
            return
            
        forecaster = RenewableEnergyForecaster(models)
    except Exception as e:
        print(f"Error initializing models: {e}")
        return
    
    # Generate forecasts for all districts
    district_forecasts = {}
    for name, info in districts.items():
        print(f"Generating forecast for {name}...")
        forecast = forecaster.forecast_district_energy(name, info)
        if forecast:
            district_forecasts[name] = forecast
    
    if not district_forecasts:
        print("Error: Failed to generate forecasts for any district.")
        return
    
    # Generate comprehensive report
    report = generate_forecast_report(district_forecasts)
    print("\nForecast Report Summary:")
    print(f"Period: {report['forecast_period']['start']} to {report['forecast_period']['end']}")
    print("\nEnergy Mix:")
    print(f"Wind: {report['state_summary']['energy_mix']['wind']:.1f}%")
    print(f"Solar: {report['state_summary']['energy_mix']['solar']:.1f}%")
    print(f"Ocean: {report['state_summary']['energy_mix']['ocean']:.1f}%")
    
    print("\nBest Districts:")
    print(f"Wind: {report['best_districts']['wind']}")
    print(f"Solar: {report['best_districts']['solar']}")
    print(f"Ocean: {report['best_districts']['ocean']}")
    
    # Visualize the forecasts
    visualize_forecasts(district_forecasts)
    
    print("\nForecast visualizations saved to disk.")

if __name__ == "__main__":
    main()