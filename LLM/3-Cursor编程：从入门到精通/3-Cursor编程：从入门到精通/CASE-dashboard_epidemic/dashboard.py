import streamlit as st
import pandas as pd
import plotly.express as px

# 读取数据
data_file = '香港各区疫情数据_20250322.xlsx'
df = pd.read_excel(data_file)

st.set_page_config(page_title="香港疫情可视化大屏", layout="wide")
st.title("香港各区疫情数据可视化大屏")

# 显示数据表和列名
def show_data_overview():
    st.subheader("数据预览")
    st.write("列名：", df.columns.tolist())
    st.dataframe(df.head())

show_data_overview()

# 假设有 '日期'、'地区'、'新增确诊'、'累计确诊' 这类列名
# 1. 确诊病例数：每日新增与累计确诊数据
if '日期' in df.columns and '新增确诊' in df.columns and '累计确诊' in df.columns:
    st.subheader("每日新增与累计确诊趋势")
    df['日期'] = pd.to_datetime(df['日期'])
    daily = df.groupby('日期').agg({'新增确诊':'sum', '累计确诊':'sum'}).reset_index()
    fig = px.line(daily, x='日期', y=['新增确诊', '累计确诊'], markers=True, labels={'value':'病例数', 'variable':'类型'})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("请检查数据中是否包含 '日期'、'新增确诊'、'累计确诊' 列")

# 2. 地理分布图：各区域疫情分布及热点区域标识
if '地区' in df.columns and '累计确诊' in df.columns:
    st.subheader("各区域累计确诊分布（地理热力图）")
    # 这里假设有地区-经纬度映射表，如无可后续补充
    # 示例：地区经纬度可用st.file_uploader或手动补充
    st.info("如需地理分布图，请补充地区经纬度信息。可上传地区-经纬度表，或在代码中补充。")
else:
    st.warning("请检查数据中是否包含 '地区'、'累计确诊' 列")

# 3. 趋势分析图：病例增长趋势，增长率变化图表
if '日期' in df.columns and '新增确诊' in df.columns:
    st.subheader("病例增长率变化趋势")
    daily = daily.sort_values('日期')
    daily['增长率'] = daily['新增确诊'].pct_change().fillna(0) * 100
    fig2 = px.bar(daily, x='日期', y='增长率', labels={'增长率':'增长率(%)'})
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("请检查数据中是否包含 '日期'、'新增确诊' 列")

st.markdown("---")
st.markdown("数据来源：香港各区疫情数据_20250322.xlsx") 