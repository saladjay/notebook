import pandas as pd

# 读取员工基本信息表
base_info = pd.read_excel('员工基本信息表.xlsx')

# 读取员工绩效表
performance = pd.read_excel('员工绩效表.xlsx')

# 筛选2024年第4季度的绩效
perf_2024_q4 = performance[(performance['年度'] == 2024) & (performance['季度'] == 4)][['员工ID', '绩效评分']]
perf_2024_q4 = perf_2024_q4.rename(columns={'绩效评分': '2024年第4季度绩效'})

# 合并到员工基本信息表
merged = pd.merge(base_info, perf_2024_q4, on='员工ID', how='left')

print('合并后的表 前5行:')
print(merged.head())

# 保存为新的Excel文件
merged.to_excel('员工信息_含2024Q4绩效.xlsx', index=False) 