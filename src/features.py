# Файл содержит спискок всех фич с описанием типа

features = {
    "train.csv":
        {"year": ("numeric" ,None),
         "station_id": ("categorical", None),
         "day": ("numeric", None),
         "ice_jam": ("numeric", None)
         },
    "hydro_1day.csv":
        {"year": ("numeric", None),
         "station_id": ("categorical", None),
         "month": ("numeric", None),
         "day": ("numeric", None),
         "date": ("drop", None),
         "stage_avg": ("numeric", None),
         "stage_min": ("numeric", None),
         "stage_max": ("numeric", None),
         "temp": ("numeric", None),
         "water_code": ("ohe", None),
         "ice_thickness": ("numeric", None),
         "snow_height": ("numeric", None),
         "place": ("categorical", None),
         "discharge": ("numeric", None)
    },
    "meteo_1day.csv":
        {"station_id": ("categorical", None),
         "year": ("numeric", None),
         "month": ("numeric", None),
         "day": ("numeric", None),
         "route_type": ("categorical", None),
         "snow_coverage_near_station": ("numeric", None),
         "snow_coverage_route": ("numeric", None),
         "ice_crust_route": ("numeric", None),
         "snow_height_aver": ("numeric", None),
         "snow_height_max": ("numeric", None),
         "snow_height_min": ("numeric", None),
         "snow_density_aver": ("numeric", None),
         "ice_crust_aver": ("numeric", None),
         "snow_saturated_thickness": ("numeric", None),
         "water_thickness": ("numeric", None),
         "water_in_snow": ("numeric", None),
         "water_total": ("numeric", None),
         "snow_coverage_charact": ("categorical", None),
         "snow_charact": ("categorical", None),
         "snow_height": ("numeric", 9999), # 9999 - means bad value
         "snow_coverage_station": ("numeric", None),
         "snow_height_q1": ("categorical", None),
         "snow_height_q2": ("categorical", None),
         "snow_height_q3": ("categorical", None),
         "temperature_20cm": ("numeric", 9999), # 9999 - means bad value
         "temperature_20cm_qual": ("categorical", None),
         "temperature_40cm": ("numeric", 9999), # 9999 - means bad value
         "temperature_40cm_qual": ("categorical", None),
         "temperature_80cm": ("numeric", 9999),  # 9999 - means bad value
         "temperature_80cm_qual": ("categorical", None),
         "temperature_120cm": ("numeric", 9999),  # 9999 - means bad value
         "temperature_120cm_qual": ("categorical", None),
         "temperature_160cm": ("numeric", 9999),  # 9999 - means bad value
         "temperature_160cm_qual": ("categorical", None),
         "temperature_240cm": ("numeric", 9999),  # 9999 - means bad value
         "temperature_240cm_qual": ("categorical", None),
         "temperature_320cm": ("numeric", 9999),  # 9999 - means bad value
         "temperature_320cm_qual": ("categorical", None),
         "temperature_ks_5cm": ("numeric", 9999),  # 9999 - means bad value
         "temperature_ks_5cm_qual": ("categorical", None),
         "temperature_ks_10cm": ("numeric", 9999),  # 9999 - means bad value
         "temperature_ks_10cm_qual": ("categorical", None),
         "temperature_ks_15cm": ("numeric", 9999),  # 9999 - means bad value
         "temperature_ks_15cm_qual": ("categorical", None),
         "temperature_ks_20cm": ("numeric", 9999),  # 9999 - means bad value
         "temperature_ks_20cm_qual": ("categorical", None),
         "date": ("drop", None),
    },
    "hydro_coord.csv":
     {
      "station_id": ("categorical", None),
      "name": ("drop", None),
      "lat": ("numeric", None),
      "lon": ("numeric", None),
      "distance_from_source": ("numeric", None, "log"),
      "drainage_area": ("numeric", None, "log"),
      "z_null": ("numeric", None, "log")
    },
    "meteo_coord.csv":
     {
      "station_id": ("categorical", None),
      "name": ("drop", None),
      "lat": ("numeric", None),
      "lon": ("numeric", None),
      "z": ("numeric", None, "log")
     }
}