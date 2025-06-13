# python convert_to_json.py

import pandas as pd
import json

# CSVデータを読み込む
df = pd.read_csv('data.csv')

# 必要なカラムだけを選択
df = df[['open', 'high', 'low', 'close']]

# 後ろの600行だけを抽出
df = df.tail(600)

# データを辞書のリストに変換
data = df.to_dict('records')

# JSON形式に変換
json_data = {
    "symbol": "EURCAD",
    "data": data
}

# JSONファイルに書き込む
with open('test_data.json', 'w') as f:
    json.dump(json_data, f, indent=2) 
