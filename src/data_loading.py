import joblib
import pandas as pd
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder, StandardScaler  # feature_range=(-1,1)

from src import DATA_DIR, TRACK1_DIR
from src.features import features

# Подтянем ближайшую к гидростанции метеостанцию


def make_water_state_encoder(base_path=TRACK1_DIR, postfix=""):
    wdf = pd.read_csv(base_path / ("reference_water_codes" + postfix + ".csv"))
    labels = wdf["water_code"].values
    le = LabelEncoder()
    le.fit(labels)
    joblib.dump(le, DATA_DIR / "dict_water_codes.joblib")

    return len(list(le.classes_))


def apply_water_state_encoder(df):
    le = joblib.load(DATA_DIR / "dict_water_codes.joblib")
    # make dict from label encoder
    # ref code -> int code
    rev_mapping = dict(zip(range(len(le.classes_)), le.classes_))
    n = len(list(le.classes_))

    # make new columns
    for i in range(n):
        df["fixed_water_code_" + str(i)] = (
            df["water_code"]
            .fillna("")
            .str.split(",")
            .apply(lambda s: str(rev_mapping[i]) in s if s else -1)
            .astype("int32")
        )

    df = df.drop(columns=["water_code"])

    return df


def fix_column(df, column, column_type, nan_encoding):
    if column_type == "numeric":
        df[f"fixed_{column}_{column_type}"] = df[column]
        df[f"fixed_{column}_{column_type}"] = df[f"fixed_{column}_{column_type}"].astype("float32")
        if nan_encoding:
            df[f"fixed_{column}_{column_type}"] = df[f"fixed_{column}_{column_type}"].replace(nan_encoding, None)
        df[f"fixed_{column}_{column_type}"] = df[f"fixed_{column}_{column_type}"].fillna(
            df[f"fixed_{column}_{column_type}"].median()
        )

    elif column_type == "categorical":
        df[f"fixed_{column}_{column_type}"] = df[column]
        if nan_encoding:
            df[f"fixed_{column}_{column_type}"] = df[f"fixed_{column}_{column_type}"].replace(nan_encoding, None)
        df[f"fixed_{column}_{column_type}"] = df[f"fixed_{column}_{column_type}"].fillna(-1)
        df[f"fixed_{column}_{column_type}"] = df[f"fixed_{column}_{column_type}"].astype("category")

    elif column_type == "water_codes":
        apply_water_state_encoder(df)

    elif column_type == "drop":
        pass

    elif column_type == "date":
        df[f"fixed_{column}_{column_type}"] = df[column]
        df[f"fixed_{column}_{column_type}"] = pd.to_datetime(df[f"fixed_{column}_{column_type}"])


def fix_df(df, df_name):

    df_features = features[df_name]
    columns = df.columns.copy()
    for column in columns:
        for column_type, nan_encoding in df_features[column]:
            fix_column(df=df, column=column, column_type=column_type, nan_encoding=nan_encoding)
        df = df.drop(columns=[column])

    assert not df.isna().any().all()

    return df


def load_table(name):
    table = pd.read_csv(TRACK1_DIR / name)
    fixed_table = fix_df(table, name)

    return fixed_table


def merge_closest_meteo_to_hydro(hydro, meteo):
    assert len(hydro) == hydro["hydro_fixed_station_id_categorical"].nunique()
    assert len(meteo) == meteo["meteo_fixed_station_id_categorical"].nunique()

    hydro["hydro_closest_meteo_station_id"] = 0
    for i, row in hydro.iterrows():
        hydro_point = row["hydro_lat_lon"]

        min_dist = float("inf")
        min_j = 0
        for j, row in meteo.iterrows():
            meteo_point = row["meteo_lat_lon"]
            if geodesic(hydro_point, meteo_point).km < min_dist:
                min_dist = geodesic(hydro_point, meteo_point).km
                min_j = j

        hydro.loc[i, "hydro_closest_meteo_station_id"] = meteo.iloc[min_j]["meteo_fixed_station_id_categorical"]

    assert (hydro["hydro_closest_meteo_station_id"] != 0).all()

    hydro = hydro.merge(meteo, left_on="hydro_closest_meteo_station_id", right_on="meteo_fixed_station_id_categorical")

    return hydro


def merge_tables(hydro_1day, meteo_1day, hydro_coord, meteo_coord):
    hydro_coord = hydro_coord.add_prefix("hydro_")
    meteo_coord = meteo_coord.add_prefix("meteo_")
    hydro_1day = hydro_1day.add_prefix("hydro_")
    meteo_1day = meteo_1day.add_prefix("meteo_")

    hydro_coord["hydro_lat_lon"] = hydro_coord[["hydro_fixed_lat_numeric", "hydro_fixed_lon_numeric"]].apply(
        tuple, axis=1
    )
    meteo_coord["meteo_lat_lon"] = meteo_coord[["meteo_fixed_lat_numeric", "meteo_fixed_lon_numeric"]].apply(
        tuple, axis=1
    )

    print("Merging closest stations")
    hydro_coord_with_closest_meteo_coord = merge_closest_meteo_to_hydro(hydro_coord, meteo_coord)

    hydro_1day = hydro_1day.merge(
        hydro_coord_with_closest_meteo_coord[["hydro_fixed_station_id_categorical", "hydro_closest_meteo_station_id"]]
    )

    hydro_1day = hydro_1day.merge(
        meteo_1day,
        left_on=["hydro_closest_meteo_station_id", "hydro_fixed_date_date"],
        right_on=["meteo_fixed_station_id_categorical", "meteo_fixed_date_date"],
    )

    hydro_1day = hydro_1day.merge(hydro_coord_with_closest_meteo_coord, on="hydro_fixed_station_id_categorical")

    return hydro_1day


def add_keys(features_df):
    features_df["year"] = features_df["hydro_fixed_year_categorical"]
    features_df["day"] = features_df["hydro_fixed_day_categorical"]
    features_df["hydro_station_id"] = features_df["hydro_fixed_station_id_categorical"]

    return features_df


def scale_numerical_features(features_df):
    numerical_cols = [col for col in features_df.columns if col.endswith("_numeric")]
    scaler = StandardScaler()
    features_df[numerical_cols] = scaler.fit_transform(features_df[numerical_cols])

    return features_df


def encode_categorical_features(features_df):
    categorical_cols = [col for col in features_df.columns if col.endswith("_categorical")]
    for cat_col in categorical_cols:
        encoder = LabelEncoder()
        features_df[cat_col] = encoder.fit_transform(features_df[cat_col])

    return features_df
