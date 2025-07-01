# 股票预测分析系统 - 完整版
# 适用于Jupyter Notebook和网页展示

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import warnings
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import tushare as ts
import os

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置 Tushare Token
ts.set_token('4c16742e8a98b00a7308d158e35e43a05eb0b5d2f8874bcf9549520b')
pro = ts.pro_api()

# 预测参数
PREDICTION_DAYS = 5
MAX_DEPTH = 6
LEARNING_RATE = 0.1
N_ESTIMATORS = 100

# 沪深300常用股票代码-名称映射
HS300_NAME_MAP = {
    "600519.SH": "贵州茅台",
    "000651.SZ": "格力电器", 
    "600036.SH": "招商银行",
    "000858.SZ": "五粮液",
    "002415.SZ": "海康威视",
    "600276.SH": "恒瑞医药",
    "000002.SZ": "万科A",
    "600887.SH": "伊利股份",
    "000568.SZ": "泸州老窖",
    "600030.SH": "中信证券"
}

def fetch_hs300_components(start_date='20250301', end_date='20250630'):
    """获取沪深300成分股"""
    try:
        df = pro.index_weight(index_code='000300.SH', start_date=start_date, end_date=end_date)
        print("沪深300成分股数据获取成功！")
        return df
    except Exception as e:
        print(f"沪深300成分股数据获取失败，错误信息：{e}")
        return None

def fetch_stock_data(ts_code, start_date='20200101', end_date='20250627'):
    """获取股票历史数据"""
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        print(f"{ts_code} 的行情数据获取成功！")
        return df
    except Exception as e:
        print(f"{ts_code} 的行情数据获取失败，错误信息：{e}")
        return None

def get_hs300_stocks():
    """获取沪深300股票列表"""
    df = fetch_hs300_components()
    if df is not None and not df.empty and 'con_code' in df.columns:
        if 'name' in df.columns:
            return list(zip(df['con_code'], df['name']))
        else:
            return [(code, HS300_NAME_MAP.get(code, code)) for code in df['con_code'].unique()]
    return []

def create_rf_features(df):
    """创建随机森林特征"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # 确保价格列为float类型
    for col in ['open', 'high', 'low', 'close', 'vol']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除包含NaN的行
    df.dropna(subset=['close'], inplace=True)
    
    if len(df) == 0:
        return df
    
    # 添加技术指标
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # 计算RSI
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    avg_loss = avg_loss.replace(0, 1e-10)
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    
    # 滞后特征
    for lag in [1, 2, 3, 5, 7, 10]:
        df[f'lag_{lag}'] = df['close'].shift(lag)
    
    df.dropna(inplace=True)
    return df

def rf_predict(data, steps=5):
    """随机森林预测"""
    try:
        # 准备数据
        df = create_rf_features(data)
        if df.empty or len(df) < 30:
            print("数据量不足，无法进行有效预测")
            return []
        
        # 准备特征和目标
        feature_columns = ['open', 'high', 'low', 'close', 'vol', 'MA5', 'MA10', 'MA20', 'RSI', 
                         'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_7', 'lag_10']
        features = df[feature_columns].dropna()
        target = df['close'].loc[features.index]
        
        if len(features) < 30:
            print("特征数据量不足")
            return []
        
        # 训练测试分割
        train_size = int(len(features) * 0.8)
        X_train = features[:train_size]
        y_train = target[:train_size]
        
        # 训练随机森林模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # 预测未来steps天
        predictions = []
        current_features = features.iloc[-1:].copy()
        
        for i in range(steps):
            # 预测下一天
            pred = rf_model.predict(current_features)[0]
            predictions.append(pred)
            
            # 更新特征（简化更新逻辑）
            if i < steps - 1:
                current_features['close'] = pred
                current_features['lag_1'] = pred
                # 更新移动平均（简化）
                if len(predictions) >= 5:
                    current_features['MA5'] = np.mean(predictions[-5:])
        
        return predictions
        
    except Exception as e:
        print(f"随机森林预测出错: {str(e)}")
        return []

def create_xgb_features(df):
    """创建XGBoost特征"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # 确保所有价格相关列为float类型
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除包含NaN的行
    df.dropna(subset=['Close'], inplace=True)
    
    if len(df) == 0:
        print("数据为空，无法创建特征")
        return df
    
    # 添加时间特征
    df['day'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    
    # 技术指标
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 计算RSI
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    avg_loss = avg_loss.replace(0, 1e-10)
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    
    # 计算MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 计算布林带
    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = sma + (std * 2)
    df['Bollinger_Lower'] = sma - (std * 2)
    
    # 滞后特征
    for lag in [1, 2, 3, 5, 7, 10]:
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    
    # 移动平均特征
    if 'Volume' in df.columns:
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    
    df.dropna(inplace=True)
    return df

def xgb_predict(data):
    """XGBoost预测"""
    try:
        # 确保数据不为空
        if data.empty:
            print("输入数据为空，无法预测")
            return pd.DataFrame(), (0, 0), (0, 0), None, None
        
        # 准备数据
        df = create_xgb_features(data)
        if df.empty:
            print("特征工程后数据为空，无法预测")
            return pd.DataFrame(), (0, 0), (0, 0), None, None
        
        features = df.drop(['Close'], axis=1, errors='ignore')
        # 只保留数值型特征，防止字符串类型导致报错
        features = features.select_dtypes(include=[np.number])
        if features.empty:
            print("特征为空，无法预测")
            return pd.DataFrame(), (0, 0), (0, 0), None, None
        
        target = df['Close']
        
        if len(df) < 30:
            print("数据量不足，无法进行有效预测")
            return pd.DataFrame(), (0, 0), (0, 0), None, None
        
        # 训练测试分割
        train_size = int(len(df) * 0.8)
        X_train, X_test = features[:train_size], features[train_size:]
        y_train, y_test = target[:train_size], target[train_size:]
        
        # 归一化
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_test_scaled = scaler.transform(X_test.values)
        
        # 转换为DMatrix格式
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train.values)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test.values)
        
        # 设置参数
        params = {
            'max_depth': MAX_DEPTH,
            'eta': LEARNING_RATE,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42
        }
        
        # 训练模型
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=N_ESTIMATORS,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            verbose_eval=False
        )
        
        # 评估模型
        train_pred = model.predict(dtrain)
        test_pred = model.predict(dtest)
        
        train_mse = mean_squared_error(y_train.values, train_pred)
        train_mae = mean_absolute_error(y_train.values, train_pred)
        test_mse = mean_squared_error(y_test.values, test_pred)
        test_mae = mean_absolute_error(y_test.values, test_pred)
        
        # 预测未来
        future_dates = pd.date_range(start=df.index[-1] + datetime.timedelta(days=1), periods=PREDICTION_DAYS)
        predictions = []
        feature_columns = features.columns.tolist()  # 保证特征顺序和数量一致
        current_data = df.iloc[-1].copy()
        
        for i in range(PREDICTION_DAYS):
            # 构造输入数据，保证顺序和数量与训练时一致
            input_data = current_data[feature_columns].values.reshape(1, -1)
            # 添加时间特征
            input_data[0, 0] = future_dates[i].day  # day
            input_data[0, 1] = future_dates[i].dayofweek  # day_of_week
            input_data[0, 2] = future_dates[i].dayofyear  # day_of_year
            # 归一化
            input_scaled = scaler.transform(input_data)
            # 转换为DMatrix
            dinput = xgb.DMatrix(input_scaled)
            # 预测
            pred = model.predict(dinput)[0]
            predictions.append(pred)
            # 更新当前数据
            current_data['Close'] = pred
            current_data = current_data.shift(-1).fillna(0)
            current_data['lag_1'] = pred
            # 更新移动平均
            if i >= 4:
                current_data['MA5'] = np.mean(predictions[-5:])
            if i >= 9:
                current_data['MA10'] = np.mean(predictions[-10:])
            if i >= 19:
                current_data['MA20'] = np.mean(predictions[-20:])
        
        # 创建结果数据框
        result_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted': predictions
        }).set_index('Date')
        
        return result_df, (train_mse, train_mae), (test_mse, test_mae), predictions, scaler
        
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        return pd.DataFrame(), (0, 0), (0, 0), None, None

def plot_kline(stock_code, data):
    """绘制K线图"""
    try:
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        # 准备数据
        df = data.copy()
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date')
        
        # 确保价格列为数值型
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # K线图
        ax1.plot(df.index, df['close'], 'b-', label='收盘价', linewidth=2)
        ax1.set_title(f'{stock_code} K线图')
        ax1.set_ylabel('价格 (元)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 成交量图
        if 'vol' in df.columns:
            df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
            ax2.bar(df.index, df['vol'], alpha=0.7, color='green', label='成交量')
            ax2.set_ylabel('成交量')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 设置日期格式
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('kline_plots', exist_ok=True)
        img_path = f'kline_plots/{stock_code}.png'
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ K线图已保存到: {img_path}")
        return img_path
        
    except Exception as e:
        print(f"绘制K线图出错: {str(e)}")
        return None

def plot_xgb_prediction(data, prediction, stock_name):
    """绘制XGBoost预测图"""
    plt.figure(figsize=(12, 6))
    
    # 历史数据
    plt.plot(data.index, data['Close'], 'b-', label='历史收盘价', linewidth=2)
    
    # 预测数据
    if isinstance(prediction, pd.DataFrame):
        if not prediction.empty:
            plt.plot(prediction.index, prediction['Predicted'], 'ro-', label='预测股价', linewidth=2)
            plt.plot(data.index[-1], data['Close'].iloc[-1], 'go', markersize=8, label='预测起始点')
    elif isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(prediction))
        plt.plot(future_dates, prediction, 'ro-', label='预测股价', linewidth=2)
        plt.plot(data.index[-1], data['Close'].iloc[-1], 'go', markersize=8, label='预测起始点')
    else:
        plt.plot(data.index[-1], data['Close'].iloc[-1], 'go', markersize=8, label='预测起始点')
    
    # 添加移动平均线
    if 'MA5' in data:
        plt.plot(data.index, data['MA5'], 'y-', label='5日均线', alpha=0.7)
    if 'MA10' in data:
        plt.plot(data.index, data['MA10'], 'm-', label='10日均线', alpha=0.7)
    if 'MA20' in data:
        plt.plot(data.index, data['MA20'], 'c-', label='20日均线', alpha=0.7)
    
    # 设置图表格式
    plt.title(f"{stock_name} 股价走势与预测")
    plt.xlabel("日期")
    plt.ylabel("价格 (元)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    # 使用股票代码作为文件名，避免特殊字符问题
    img_path = f"{stock_name.replace('*', '_').replace('/', '_')}_预测图表.png"
    plt.savefig(img_path)
    print(f"图表已保存为: {img_path}")
    plt.close()
    return img_path

def predict_stock(stock_code, days=180):
    """预测指定股票"""
    print(f"开始预测股票: {stock_code}")
    
    # 获取数据
    end_date = datetime.date(2025, 6, 26)
    start_date = end_date - datetime.timedelta(days=days)
    
    stock_data = fetch_stock_data(
        stock_code, 
        start_date.strftime('%Y%m%d'), 
        end_date.strftime('%Y%m%d')
    )
    
    if stock_data is None or stock_data.empty:
        print(f"无法获取{stock_code}的数据")
        return None
    
    # 准备数据格式
    if 'trade_date' in stock_data.columns:
        stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
        stock_data = stock_data.sort_values('trade_date')
        stock_data = stock_data.set_index('trade_date')
    
    # 随机森林预测
    print("使用随机森林预测...")
    rf_predictions = rf_predict(stock_data, steps=5)
    rf_img = plot_kline(stock_code, stock_data)
    
    # XGBoost预测
    print("使用XGBoost预测...")
    # 准备XGBoost数据格式
    rename_dict = {
        'open': 'Open', 'high': 'High', 'low': 'Low', 
        'close': 'Close', 'vol': 'Volume'
    }
    stock_data_for_xgb = stock_data.rename(columns={k: v for k, v in rename_dict.items() if k in stock_data.columns})
    
    xgb_result, xgb_metrics, xgb_mae_metrics, xgb_predictions, xgb_scaler = xgb_predict(stock_data_for_xgb)
    xgb_img = plot_xgb_prediction(stock_data_for_xgb, xgb_predictions, stock_code)
    
    # 返回结果
    return {
        'stock_code': stock_code,
        'rf_predictions': rf_predictions,
        'rf_image': rf_img,
        'xgb_predictions': xgb_predictions,
        'xgb_metrics': xgb_metrics,
        'xgb_mae_metrics': xgb_mae_metrics,
        'xgb_image': xgb_img
    }

def display_results(results):
    """显示预测结果"""
    if results is None:
        print("预测失败")
        return
    
    print(f"\n=== {results['stock_code']} 预测结果 ===")
    
    # 随机森林结果
    print("\n【随机森林预测】")
    if results['rf_predictions']:
        for i, pred in enumerate(results['rf_predictions']):
            print(f"第{i+1}天: {pred:.2f}元")
    else:
        print("随机森林预测失败")
    
    # XGBoost结果
    print("\n【XGBoost预测】")
    if results['xgb_predictions']:
        for i, pred in enumerate(results['xgb_predictions']):
            print(f"第{i+1}天: {pred:.2f}元")
        
        if results['xgb_metrics']:
            print(f"训练MSE: {results['xgb_metrics'][0]:.4f}")
            print(f"训练MAE: {results['xgb_metrics'][1]:.4f}")
        if results['xgb_mae_metrics']:
            print(f"测试MSE: {results['xgb_mae_metrics'][0]:.4f}")
            print(f"测试MAE: {results['xgb_mae_metrics'][1]:.4f}")
    else:
        print("XGBoost预测失败")
    
    print(f"\n图表已保存:")
    if results['rf_image']:
        print(f"随机森林K线图: {results['rf_image']}")
    if results['xgb_image']:
        print(f"XGBoost预测图: {results['xgb_image']}")

# 使用示例
if __name__ == "__main__":
    # 预测指定股票
    stock_code = "600519.SH"  # 贵州茅台
    results = predict_stock(stock_code)
    display_results(results)
    
    # 随机选择股票预测
    print("\n" + "="*50)
    print("随机选择股票预测:")
    stocks = get_hs300_stocks()
    if stocks:
        stock_code, stock_name = random.choice(stocks)
        print(f"随机选择: {stock_name} ({stock_code})")
        results = predict_stock(stock_code)
        display_results(results)
    else:
        print("无法获取随机股票") 