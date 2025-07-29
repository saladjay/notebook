import pandas as pd

# 读取 Excel 文件
file_path = '香港各区疫情数据_20250322.xlsx'
df = pd.read_excel(file_path)

# 显示列名和前几行，帮助确认数据结构
print("列名：", df.columns.tolist())
print(df.head())

# 假设有 '日期'、'地区'、'新增确诊'、'累计确诊' 这类列名
# 计算每日新增和累计确诊（如需调整列名请根据实际情况修改）
if '新增确诊' in df.columns and '累计确诊' in df.columns:
    print("\n每日新增确诊：")
    print(df[['日期', '地区', '新增确诊']].head())
    print("\n累计确诊：")
    print(df[['日期', '地区', '累计确诊']].head())
else:
    print("请根据实际列名修改代码中的列名。")