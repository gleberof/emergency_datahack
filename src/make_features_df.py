from src import DATA_DIR
from src.utils.data import add_keys, encode_categorical_features, load_table, merge_tables, scale_numerical_features

if __name__ == "__main__":
    hydro_1day = load_table("hydro_1day.csv")
    data = load_table("train.csv")
    meteo_1day = load_table("meteo_1day.csv")
    hydro_coord = load_table("hydro_coord.csv")
    meteo_coord = load_table("meteo_coord.csv")
    features_df = merge_tables(hydro_1day, meteo_1day, hydro_coord, meteo_coord)
    features_df = add_keys(features_df)
    features_df = scale_numerical_features(features_df)
    features_df = encode_categorical_features(features_df)
    bad_columns = [col for col in features_df.columns if (col.endswith("_x") or col.endswith("_y"))] + [
        "hydro_lat_lon",
        "meteo_lat_lon",
    ]
    features_df = features_df.drop(columns=bad_columns)
    features_df.to_csv(DATA_DIR / "features.csv")
    data.to_csv(DATA_DIR / "train.csv")
