import pandas as pd
import numpy as np
import math


def get_match_observatory_dict():
    gas_usage_df = pd.read_csv("./data/2005_gas_usage.csv", encoding="euc-kr")
    temperature_df = pd.read_csv("./data/2005_temperature.csv", encoding="euc-kr")
    result = dict()
    for i in gas_usage_df.index:
        area_x = gas_usage_df.loc[i, "X 좌표"]
        area_y = gas_usage_df.loc[i, "Y 좌표"]
        area_name = gas_usage_df.loc[i, "행정동명"]
        min = 99999999999999999
        for j in temperature_df.index:
            observatory_x = temperature_df.loc[j, "X 좌표"]
            observatory_y = temperature_df.loc[j, "Y 좌표"]
            observatory_name = temperature_df.loc[j, "관측소명"]
            x = area_x - observatory_x
            y = area_y - observatory_y
            distance = math.sqrt((x * x) + (y * y))
            if distance < min:
                min = distance
                result[area_name] = observatory_name
    return result


def get_merge_df(year, match_dict):
    gas_usage_df = pd.read_csv(f"./data/{year}_gas_usage.csv", encoding="euc-kr")
    temperature_df = pd.read_csv(f"./data/{year}_temperature.csv", encoding="euc-kr")
    for idx in gas_usage_df.index:
        area_name = gas_usage_df.loc[idx, "행정동명"]
        observatoty_name = match_dict[area_name]
        match_row = temperature_df[temperature_df["관측소명"] == observatoty_name]
        gas_usage_df.loc[idx, "min_temperature"] = np.array(match_row["평균최저기온"])[0]

    gas_usage_df = gas_usage_df[["행정동명", f"{year}년사용량", "min_temperature"]]
    gas_usage_df = gas_usage_df.rename(
        columns={
            "행정동명": "area_name",
            f"{year}년사용량": "gas_usage",
        }
    )
    gas_usage_df["year"] = year
    return gas_usage_df


def split_df(df):
    train_df = df[df["year"] != 2008]
    test_df = df[df["year"] == 2008]
    submission_df = test_df["gas_usage"]
    submission_df.to_csv("./data/submission.csv")
    submission_df = pd.read_csv("./data/submission.csv")
    submission_df.rename(columns={"Unnamed: 0": "id"})
    test_df = test_df.drop(["gas_usage"], axis=1)
    return train_df, test_df, submission_df


match_dict = get_match_observatory_dict()

merged_df = None
years = [2005, 2006, 2007, 2008]
for year in years:
    if merged_df is None:
        merged_df = get_merge_df(year, match_dict)
    else:
        merged_df = pd.concat([merged_df, get_merge_df(year, match_dict)], axis=0)

train_df, test_df, submission_df = split_df(merged_df)

merged_df.to_csv("./data/merged_data.csv", index=False)
train_df.to_csv("./data/train_data.csv", index=False)
test_df.to_csv("./data/test_data.csv", index=False)
submission_df.to_csv("./data/submission.csv", index=False)
