import tushare as ts
import pandas as pd

# 设置 Tushare Token
ts.set_token('4c16742e8a98b00a7308d158e35e43a05eb0b5d2f8874bcf9549520b')  # 替换为你的 Token
pro = ts.pro_api()


# 获取沪深300成分股权重列表
def fetch_hs300_components(start_date='20250301', end_date='20250630'):
    try:
        df = pro.index_weight(index_code='000300.SH', start_date=start_date, end_date=end_date)
        print("沪深300成分股数据获取成功！")
        return df
    except Exception as e:
        print(f"沪深300成分股数据获取失败，错误信息：{e}")
        return None


# 获取成分股的历史日线行情数据
def fetch_stock_data(ts_code, start_date='20200101', end_date='20250701'):
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        print(f"{ts_code} 的行情数据获取成功！")
        return df
    except Exception as e:
        print(f"{ts_code} 的行情数据获取失败，错误信息：{e}")
        return None


# 获取成分股的财务数据  利润表、资产负债表、现金流量表
def fetch_stock_financial_data(ts_code, start_date='20200101', end_date='20250701'):
    try:
        df_income = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df_balance = pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df_cashflow = pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=end_date)

        df = pd.merge(df_income, df_balance, on='end_date', how='left', suffixes=('_income', '_balance'))
        df = pd.merge(df, df_cashflow, on='end_date', how='left', suffixes=('', '_cashflow'))

        print(f"{ts_code} 的财务数据获取成功！")
        return df
    except Exception as e:
        print(f"{ts_code} 的财务数据获取失败，错误信息：{e}")
        return None


# 保存数据到 CSV 文件
def save_data_to_csv(df, file_name):
    if df is not None and not df.empty:
        df.to_csv(file_name, index=False, encoding='utf-8')
        print(f"数据已保存到 {file_name}")
    else:
        print(f"没有数据可保存到 {file_name}")


# 沪深300常用股票代码-名称映射（可补充）
HS300_NAME_MAP = {
    "600519.SH": "贵州茅台",
    "000651.SZ": "格力电器",
    "600036.SH": "招商银行",
    # ...可补充更多
}

def get_hs300_stocks():
    df = fetch_hs300_components()
    if df is not None and not df.empty and 'con_code' in df.columns:
        if 'name' in df.columns:
            return list(zip(df['con_code'], df['name']))
        else:
            # 没有name列时用静态字典补全，否则用代码本身
            return [(code, HS300_NAME_MAP.get(code, code)) for code in df['con_code'].unique()]
    return []


if __name__ == "__main__":
    # 获取沪深300成分股权重列表
    components = fetch_hs300_components()
    if components is not None:
        save_data_to_csv(components, 'hs300_components.csv')

    # 获取每个成分股的数据
    for index, row in components.iterrows():
        ts_code = row['con_code']
        stock_data = fetch_stock_data(ts_code)
        financial_data = fetch_stock_financial_data(ts_code)

        if stock_data is not None:
            save_data_to_csv(stock_data, f'stock_data_{ts_code}.csv')
        if financial_data is not None:
            save_data_to_csv(financial_data, f'financial_data_{ts_code}.csv')