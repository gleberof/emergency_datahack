import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # feature_range=(-1,1)

from src import DATA_DIR, TRACK1_DIR
from src.features import features


def make_water_state_encoder(base_path=TRACK1_DIR, postfix=""):
    wdf = pd.read_csv(TRACK1_DIR / ("reference_water_codes" + postfix + ".csv"))
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
        df["water_code_" + str(i)] = (
            df["water_code"]
            .fillna("")
            .str.split(",")
            .apply(lambda s: str(rev_mapping[i]) in s if s else -1)
            .astype(int)
        )
    return df


def make_hydro_df(base_path=TRACK1_DIR, postfix=""):
    df = pd.read_csv(base_path / ("hydro_1day" + postfix + ".csv"))
    df = apply_water_state_encoder(df)

    # split features
    df["cat_year"] = df["year"]
    df["cat_month"] = df["month"]
    df["cat_day"] = df["day"]
    cat_feats = ["cat_year", "cat_month", "cat_day"]
    num_feats, drop_feats = [], []
    for c, t in features["hydro_1day.csv"].items():
        if t == "drop":
            drop_feats.append(c)
        elif t == "numeric":
            num_feats.append(c)
        elif t == "categorical":
            cat_feats.append(c)
    target_feats = []
    for c in df.columns:
        if c.startswith("water_code_"):
            target_feats.append(c)

    # numerical features
    num_scaler = MinMaxScaler(feature_range=(-1, 1))
    df[num_feats] = num_scaler.fit_transform(df[num_feats])
    joblib.dump(num_scaler, DATA_DIR / "num_scaler.joblib")

    # categorical features
    for c in cat_feats:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        joblib.dump(le, DATA_DIR / (c + "_le.joblib"))

    return df[num_feats + cat_feats + target_feats], [num_feats, cat_feats, target_feats]
