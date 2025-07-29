from flask import Flask, jsonify, render_template
import pandas as pd

app = Flask(__name__)

df = pd.read_excel('香港各区疫情数据_20250322.xlsx')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/overview')
def overview():
    df['日期'] = pd.to_datetime(df['报告日期'])
    daily = df.groupby('日期').agg({'新增确诊':'sum', '累计确诊':'sum'}).reset_index()
    daily['日期'] = daily['日期'].dt.strftime('%Y-%m-%d')
    return jsonify(daily.to_dict(orient='records'))

@app.route('/api/region')
def region():
    region = df.groupby('地区名称').agg({'累计确诊':'max'}).reset_index()
    return jsonify(region.to_dict(orient='records'))

@app.route('/api/growth')
def growth():
    df['日期'] = pd.to_datetime(df['报告日期'])
    daily = df.groupby('日期').agg({'新增确诊':'sum'}).reset_index()
    daily = daily.sort_values('日期')
    daily['增长率'] = daily['新增确诊'].pct_change().fillna(0) * 100
    daily['日期'] = daily['日期'].dt.strftime('%Y-%m-%d')
    return jsonify(daily[['日期', '增长率']].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True) 