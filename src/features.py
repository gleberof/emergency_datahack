# Файл содержит спискок всех фич с описанием типа

features = {
    "train.csv": {"year": "numeric", "station_id": "categorical", "day": "numeric", "ice_jam": "numeric"},
    "hydro_1day.csv": {
        "year": "numeric",
        "station_id": "categorical",
        "month": "numeric",
        "day": "numeric",
        "date": "drop",
        "stage_avg": "numeric",
        "stage_min": "numeric",
        "stage_max": "numeric",
        "temp": "numeric",
        "water_code": "ohe",
        "ice_thickness": "numeric",
        "snow_height": "numeric",
        "place": "categorical",
        "discharge": "numeric",
    },
    "meteo_1day.csv": {
        "station_id": "categorical",
        "year": "numeric",
        "month": "numeric",
        "day": "numeric",
        "route_type": "categorical",
        "snow_coverage_near_station": "numeric",
        "snow_coverage_route": "numeric",
        "ice_crust_route": "numeric",
        "snow_height_aver": "numeric",
        "snow_height_max": "numeric",
        "snow_height_min": "numeric",
        "snow_density_aver": "numeric",
        "ice_crust_aver": "numeric",
        "snow_saturated_thickness": "numeric",
        "water_thickness": "numeric",
        "water_in_snow": "numeric",
        "water_total": "numeric",
        "snow_coverage_charact": "categorical",
        "snow_charact": "categorical",
        "snow_height": "numeric",  # 9999 - means bad value
        "snow_coverage_station": "numeric",
        "snow_height_q1": "categorical",
        "snow_height_q2": "categorical",
        "snow_height_q3": "categorical",
        "temperature_20cm": "numeric",  # 9999 - means bad value
        "temperature_20cm_qual": "categorical",
        "temperature_40cm": "numeric",  # 9999 - means bad value
        "temperature_40cm_qual": "categorical",
        "temperature_80cm": "numeric",  # 9999 - means bad value
        "temperature_80cm_qual": "categorical",
        "temperature_120cm": "numeric",  # 9999 - means bad value
        "temperature_120cm_qual": "categorical",
        "temperature_160cm": "numeric",  # 9999 - means bad value
        "temperature_160cm_qual": "categorical",
        "temperature_240cm": "numeric",  # 9999 - means bad value
        "temperature_240cm_qual": "categorical",
        "temperature_320cm": "numeric",  # 9999 - means bad value
        "temperature_320cm_qual": "categorical",
        "temperature_ks_5cm": "numeric",  # 9999 - means bad value
        "temperature_ks_5cm_qual": "categorical",
        "temperature_ks_10cm": "numeric",  # 9999 - means bad value
        "temperature_ks_10cm_qual": "categorical",
        "temperature_ks_15cm": "numeric",  # 9999 - means bad value
        "temperature_ks_15cm_qual": "categorical",
        "temperature_ks_20cm": "numeric",  # 9999 - means bad value
        "temperature_ks_20cm_qual": "categorical",
        "date": "drop",
    },
}
