import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
import mplfinance as mpf
import argparse
import sys

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存图表的目录
os.makedirs('kline_plots', exist_ok=True)

# 计算 RSI
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# 加载数据
def load_stock_data(stock_data_file, financial_data_file):
    try:
        stock_data = pd.read_csv(stock_data_file)
        financial_data = pd.read_csv(financial_data_file)
        print(f"✅ 数据加载成功：{stock_data_file} 和 {financial_data_file}")
        return stock_data, financial_data
    except Exception as e:
        print(f"❌ 数据加载失败，错误信息：{e}")
        return None, None

# 特征工程与合并数据
def merge_stock_data(stock_data, financial_data, ts_code):
    try:
        required_columns = ['open', 'high', 'low', 'close', 'vol', 'amount']
        for col in required_columns:
            if col not in stock_data.columns:
                stock_data[col] = 0  # 或者采取其他补救措施

        stock_data.rename(columns={'trade_date': 'date'}, inplace=True)
        financial_data.rename(columns={'end_date': 'date'}, inplace=True)

        stock_data['date'] = pd.to_datetime(stock_data['date'])
        financial_data['date'] = pd.to_datetime(financial_data['date'])

        # 技术指标
        stock_data['MA5'] = stock_data['close'].rolling(window=5).mean()
        stock_data['MA10'] = stock_data['close'].rolling(window=10).mean()
        stock_data['MA20'] = stock_data['close'].rolling(window=20).mean()
        stock_data['RSI'] = calculate_rsi(stock_data['close'])
        stock_data['turnover_rate'] = (stock_data['vol'] / stock_data['amount']).fillna(0)
        stock_data['volatility'] = stock_data['close'].pct_change().rolling(window=20).std().fillna(0)

        # MACD
        stock_data['EMA12'] = stock_data['close'].ewm(span=12, adjust=False).mean()
        stock_data['EMA26'] = stock_data['close'].ewm(span=26, adjust=False).mean()
        stock_data['MACD'] = (stock_data['EMA12'] - stock_data['EMA26']).fillna(0)
        stock_data['Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean().fillna(0)
        stock_data['Hist'] = (stock_data['MACD'] - stock_data['Signal']).fillna(0)
        stock_data.drop(columns=['EMA12', 'EMA26'], inplace=True)

        # 删除缺失行
        stock_data.dropna(
            subset=['MA5', 'MA10', 'MA20', 'RSI', 'turnover_rate', 'volatility', 'MACD', 'Signal', 'Hist'],
            inplace=True)

        merged_data = pd.merge(stock_data, financial_data[['date', 'revenue', 'net_profit']],
                               on='date', how='left')

        if 'revenue' in merged_data.columns:
            merged_data['revenue_growth'] = merged_data['revenue'].pct_change().fillna(0)
        if 'net_profit' in merged_data.columns:
            merged_data['net_profit_growth'] = merged_data['net_profit'].pct_change().fillna(0)

        merged_data.to_csv(f'merged_data_{ts_code}.csv', index=False)
        print(f"📁 整合后的数据已保存到 merged_data_{ts_code}.csv")
        return merged_data
    except Exception as e:
        print(f"❌ 数据整合失败，错误信息：{e}")
        return None

# 构建多步目标（未来5天收盘价）
def create_multi_step_target(data, steps=5, target_col='close'):
    try:
        # 保证有close列
        if target_col not in data.columns:
            if target_col.capitalize() in data.columns:
                data[target_col] = data[target_col.capitalize()]
            else:
                raise KeyError(f"数据中没有'{target_col}'或'{target_col.capitalize()}'列，请检查数据格式！")
        # 保证close列为float
        data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
        
        # 特征工程
        data = data.copy()
        
        # 技术指标
        data['MA5'] = data[target_col].rolling(window=5).mean()
        data['MA10'] = data[target_col].rolling(window=10).mean()
        data['MA20'] = data[target_col].rolling(window=20).mean()
        data['RSI'] = calculate_rsi(data[target_col])
        data['volatility'] = data[target_col].pct_change().rolling(window=20).std().fillna(0)
        
        # MACD
        data['EMA12'] = data[target_col].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data[target_col].ewm(span=26, adjust=False).mean()
        data['MACD'] = (data['EMA12'] - data['EMA26']).fillna(0)
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean().fillna(0)
        data['Hist'] = (data['MACD'] - data['Signal']).fillna(0)
        
        # 滞后特征
        for i in range(1, 6):
            data[f'lag_{i}'] = data[target_col].shift(i)
        
        # 构建多步目标
        targets = []
        for i in range(steps):
            data[f'target_{i + 1}'] = data[target_col].shift(-i - 1)
            targets.append(f'target_{i + 1}')
        
        # 删除缺失值
        data.dropna(inplace=True)
        
        if len(data) < 30:
            print("数据量不足，无法进行有效预测")
            return [0.0] * steps
        
        # 特征列
        features = ['MA5', 'MA10', 'MA20', 'RSI', 'volatility', 'MACD', 'Signal', 'Hist'] + [f'lag_{i}' for i in range(1, 6)]
        features = [f for f in features if f in data.columns]
        
        if len(features) == 0:
            print("没有可用的特征列")
            return [0.0] * steps
        
        # 准备训练数据
        X = data[features]
        y = data[targets].values
        
        # 数据填充
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # 训练测试分割
        train_size = int(len(X_imputed) * 0.8)
        X_train, X_test = X_imputed[:train_size], X_imputed[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 训练随机森林模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 预测
        latest_data = data[features].iloc[-1].values.reshape(1, -1)
        latest_data_imputed = imputer.transform(latest_data)
        predicted_prices = model.predict(latest_data_imputed)[0]
        
        # 返回预测价格列表
        return [float(price) for price in predicted_prices]
        
    except Exception as e:
        print(f"随机森林预测失败: {e}")
        return [0.0] * steps

# 绘制 K 线图
def plot_kline(ts_code, data):
    try:
        # 如果没有date列，用index
        if 'date' not in data.columns:
            data = data.reset_index()
            data['date'] = data.index
        # 确保有必要的列
        required_cols = ['open', 'high', 'low', 'close', 'vol']
        for col in required_cols:
            if col not in data.columns:
                if col.capitalize() in data.columns:
                    data[col] = data[col.capitalize()]
                else:
                    raise KeyError(f"数据中没有'{col}'列，请检查数据格式！")
        # 重命名列以符合mplfinance要求
        df = data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'vol': 'Volume',
            'date': 'Date'
        })
        # 设置日期为索引
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        # 创建保存目录
        save_dir = 'kline_plots'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{ts_code}.png")
        # 绘图并保存
        mpf.plot(df, type='candle', style='charles', title=f'{ts_code} K线图', ylabel='价格',
                volume=True, savefig=save_path, figsize=(10, 6))
        print(f"✅ K线图已保存到: {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ 绘制 {ts_code} K线图失败，错误信息：{e}")
        return None

# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('stock_code', nargs='?', type=str, help='股票代码')
    parser.add_argument('--stock_code', type=str, dest='stock_code_named', help='股票代码（命名参数）')
    args = parser.parse_args()
    
    # 优先使用位置参数，如果没有则使用命名参数
    stock_code = args.stock_code or args.stock_code_named
    
    if stock_code:
        # 单只股票预测
        print(f"只预测单只股票: {stock_code}")
        
        # 使用data_fetcher获取数据
        try:
            from data_fetcher import fetch_stock_data
            import datetime
            
            end_date = datetime.date(2025, 6, 26)
            start_date = end_date - datetime.timedelta(days=180)
            
            # 获取股票数据
            stock_data = fetch_stock_data(stock_code, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
            
            if stock_data is None or stock_data.empty:
                print(f"❌ 无法获取{stock_code}的数据")
                sys.exit(1)
            
            # 处理数据格式
            if 'trade_date' in stock_data.columns:
                stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
                stock_data = stock_data.sort_values('trade_date')
                stock_data = stock_data.set_index('trade_date')
                stock_data = stock_data.rename(columns={'trade_date': 'date'})
            
            # 确保列名正确
            rename_dict = {
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'vol': 'vol'
            }
            for orig, new in rename_dict.items():
                if orig in stock_data.columns:
                    stock_data[new] = stock_data[orig]
            
            # 确保价格列为float类型
            for col in ['close', 'open', 'high', 'low']:
                if col in stock_data.columns:
                    stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
            
            # 创建空的财务数据（随机森林不需要）
            financial_data = pd.DataFrame({
                'date': stock_data.index,
                'revenue': [0] * len(stock_data),
                'net_profit': [0] * len(stock_data)
            })
            
            # 合并数据
            merged_data = merge_stock_data(stock_data.reset_index(), financial_data, stock_code)
            
            if merged_data is not None:
                # 确保merged_data有正确的索引
                if 'trade_date' in merged_data.columns:
                    merged_data['trade_date'] = pd.to_datetime(merged_data['trade_date'])
                    merged_data = merged_data.sort_values('trade_date')
                    merged_data = merged_data.set_index('trade_date')
                elif 'date' in merged_data.columns:
                    merged_data['date'] = pd.to_datetime(merged_data['date'])
                    merged_data = merged_data.sort_values('date')
                    merged_data = merged_data.set_index('date')
                
                # 进行预测
                predicted_prices = create_multi_step_target(merged_data, steps=5)
                
                # 保存预测结果
                result_df = pd.DataFrame({
                    'date': pd.date_range(start=merged_data.index[-1] + pd.Timedelta(days=1), periods=5),
                    'predicted_price': predicted_prices,
                    'model': 'Random Forest'
                })
                result_df.to_csv(f'{stock_code}_随机森林预测结果.csv', index=False)
                print(f"随机森林预测结果已保存到: {stock_code}_随机森林预测结果.csv")
                
                # 绘制K线图
                plot_kline(stock_code, merged_data)
                
                print(f"✅ {stock_code} 随机森林预测完成！")
            else:
                print(f"❌ {stock_code} 数据处理失败")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ 预测失败，错误信息：{e}")
            sys.exit(1)
    else:
        # 批量预测所有沪深300成分股
        print("批量预测所有沪深300成分股...")
        components = pd.read_csv('hs300_components.csv')
        all_predictions = []

        features = ['MA5', 'MA10', 'MA20', 'RSI', 'turnover_rate', 'volatility',
                    'MACD', 'Signal', 'Hist', 'revenue_growth', 'net_profit_growth']

        for index, row in components.iterrows():
            ts_code = row['con_code']
            try:
                # 在线拉取行情数据
                from data_fetcher import fetch_stock_data
                import datetime
                end_date = datetime.date(2025, 6, 26)
                start_date = end_date - datetime.timedelta(days=180)
                stock_data = fetch_stock_data(ts_code, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
                if stock_data is None or stock_data.empty:
                    print(f"❌ 无法获取{ts_code}的数据")
                    continue
                # 处理数据格式
                if 'trade_date' in stock_data.columns:
                    stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
                    stock_data = stock_data.sort_values('trade_date')
                    stock_data = stock_data.set_index('trade_date')
                    stock_data = stock_data.rename(columns={'trade_date': 'date'})
                # 确保列名正确
                rename_dict = {
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'vol': 'vol'
                }
                for orig, new in rename_dict.items():
                    if orig in stock_data.columns:
                        stock_data[new] = stock_data[orig]
                for col in ['close', 'open', 'high', 'low']:
                    if col in stock_data.columns:
                        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                # 创建空的财务数据（随机森林不需要）
                financial_data = pd.DataFrame({
                    'date': stock_data.index,
                    'revenue': [0] * len(stock_data),
                    'net_profit': [0] * len(stock_data)
                })
                # 合并数据
                merged_data = merge_stock_data(stock_data.reset_index(), financial_data, ts_code)
                if merged_data is not None:
                    if 'trade_date' in merged_data.columns:
                        merged_data['trade_date'] = pd.to_datetime(merged_data['trade_date'])
                        merged_data = merged_data.sort_values('trade_date')
                        merged_data = merged_data.set_index('trade_date')
                    elif 'date' in merged_data.columns:
                        merged_data['date'] = pd.to_datetime(merged_data['date'])
                        merged_data = merged_data.sort_values('date')
                        merged_data = merged_data.set_index('date')
                    predicted_prices = create_multi_step_target(merged_data, steps=5)
                    all_predictions.append({
                        'ts_code': ts_code,
                        'pred_next_5': predicted_prices
                    })
                    print(f"{ts_code} 预测未来5天收盘价: {predicted_prices}")
                    # 保存每只股票的预测结果到独立CSV
                    result_df = pd.DataFrame({
                        'date': pd.date_range(start=merged_data.index[-1] + pd.Timedelta(days=1), periods=5),
                        'predicted_price': predicted_prices,
                        'model': 'Random Forest'
                    })
                    result_df.to_csv(f'{ts_code}_随机森林预测结果.csv', index=False)
                    print(f"随机森林预测结果已保存到: {ts_code}_随机森林预测结果.csv")
                    plot_kline(ts_code, merged_data)
                else:
                    print(f"❌ {ts_code} 数据处理失败")
            except Exception as e:
                print(f"{ts_code} 数据处理失败，错误信息：{e}")

        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv('all_hsz300_predictions_5days.csv', index=False)
        print("✅ 所有沪深300成分股预测完成，结果已保存至 all_hsz300_predictions_5days.csv")