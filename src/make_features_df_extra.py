from src import DATA_DIR, TRACK1_DIR, TRACK1_EXTRA_DIR
from src.utils.data import (
    add_keys,
    encode_categorical_features,
    load_table,
    make_water_state_encoder,
    merge_tables,
    scale_numerical_features,
)

if __name__ == "__main__":
    make_water_state_encoder(TRACK1_DIR)
    hydro_1day = load_table(TRACK1_EXTRA_DIR, "hydro_1day.csv")
    meteo_1day = load_table(TRACK1_EXTRA_DIR, "meteo_1day.csv")
    hydro_coord = load_table(TRACK1_DIR, "hydro_coord.csv")
    meteo_coord = load_table(TRACK1_DIR, "meteo_coord.csv")
    features_df = merge_tables(hydro_1day, meteo_1day, hydro_coord, meteo_coord)
    features_df = add_keys(features_df)
    features_df = scale_numerical_features(features_df)
    features_df = encode_categorical_features(features_df)
    features_df.to_csv(DATA_DIR / "features_extra.csv", index=False)
