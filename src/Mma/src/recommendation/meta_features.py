import requests,os
import datetime
from timezonefinder import TimezoneFinder
from pytz import timezone

class MetaFeatureExtractor:
    def __init__(self, weather_api_key):
        self.weather_api_key = weather_api_key
        self.tf = TimezoneFinder()

    def get_public_ip(self):
        """Fetch the user's public IP address dynamically."""
        try:
            response = requests.get("https://api64.ipify.org?format=json")
            return response.json().get("ip", "unknown")
        except Exception:
            return "unknown"

    def get_location(self, ip_address):
        """Fetch the geolocation data for the given IP address."""
        try:
            response = requests.get(f"https://ipinfo.io/{ip_address}/json")
            data = response.json()
            city = data.get("city", "unknown")
            country = data.get("country", "unknown")
            loc = data.get("loc", "0,0").split(",")
            latitude, longitude = float(loc[0]), float(loc[1])
            return city, country, latitude, longitude
        except Exception:
            return "unknown", "unknown", 0, 0

    def get_weather(self, latitude, longitude):
        """Fetch current weather conditions using OpenWeather API."""
        if latitude == 0 and longitude == 0:
            return {"temperature": "unknown", "weather_condition": "unknown", "humidity": "unknown"}

        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            return {
                "temperature": data["main"]["temp"],
                "weather_condition": data["weather"][0]["main"],
                "humidity": data["main"]["humidity"]
            }
        except Exception:
            return {"temperature": "unknown", "weather_condition": "unknown", "humidity": "unknown"}

    def get_time_features(self, latitude, longitude):
        """Extract time-based features using the user's timezone."""
        try:
            timezone_str = self.tf.timezone_at(lat=latitude, lng=longitude)
            if not timezone_str:
                return {"hour": "unknown", "day_of_week": "unknown", "month": "unknown", "is_weekend": "unknown", "time_of_day": "unknown"}
            
            now = datetime.datetime.now(timezone(timezone_str))
            hour = now.hour
            day_of_week = now.strftime("%A")  # Human-readable day name
            month = now.month
            is_weekend = day_of_week in ["Saturday", "Sunday"]
            time_of_day = self._classify_time_of_day(hour)

            return {
                "hour": hour,
                "day_of_week": day_of_week,  # Now correctly displayed as a string
                "month": month,
                "is_weekend": is_weekend,
                "time_of_day": time_of_day
            }
        except Exception:
            return {"hour": "unknown", "day_of_week": "unknown", "month": "unknown", "is_weekend": "unknown", "time_of_day": "unknown"}

    def _classify_time_of_day(self, hour):
        """Classify time into morning, afternoon, evening, or night."""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def get_all_meta_features(self):
        """Fetch all meta features including IP-based location, weather, and time attributes."""
        ip_address = self.get_public_ip()
        city, country, latitude, longitude = self.get_location(ip_address)
        weather_data = self.get_weather(latitude, longitude)
        time_features = self.get_time_features(latitude, longitude)

        meta_features = {
            "ip_address": ip_address,
            "city": city,
            "country": country,
            "latitude": latitude,
            "longitude": longitude,
            **weather_data,
            **time_features
        }

        return meta_features
