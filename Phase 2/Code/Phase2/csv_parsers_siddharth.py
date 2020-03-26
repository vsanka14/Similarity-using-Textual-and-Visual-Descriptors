#upadted csv parsers to include location names as an additional column
import os
import pandas as pd

def parse_csv(location_dict, model):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filepath = '../descvis/img/'
    all_data = []
    for location in location_dict:
        df = pd.read_csv("../descvis/img/" + location + " " + model + ".csv", header = None) 
        df.rename(columns={0: 'img_id'}, inplace=True) 
        df.set_index('img_id', inplace=True, drop=True)
        locations = [location]*len(df)
        df["location_name"]=locations
        all_data.append(df)
    csv_df = pd.concat(all_data)
    print(csv_df.head())
    return csv_df
