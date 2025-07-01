import pandas as pd

# 读取原始文件
df = pd.read_csv('hs300_components.csv')

# 按股票代码去重，保留权重最高的记录
df_clean = df.drop_duplicates(subset=['con_code'], keep='first')

# 按权重降序排序
df_clean = df_clean.sort_values('weight', ascending=False)

# 只保留前300个成分股
df_clean = df_clean.head(300)

# 删除trade_date列
df_clean = df_clean.drop('trade_date', axis=1)

# 保存清理后的文件
df_clean.to_csv('hs300_components_clean.csv', index=False)

print(f"清理完成！")
print(f"原始文件行数: {len(df)}")
print(f"去重后行数: {len(df_clean)}")
print(f"前10个成分股:")
print(df_clean[['con_code', 'weight']].head(10)) 