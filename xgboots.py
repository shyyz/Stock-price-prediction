import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from data_fetcher import fetch_hs300_components, fetch_stock_data  # 导入数据获取模块
import argparse
import warnings

# 忽略matplotlib的字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 设置matplotlib不显示警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 固定参数值
PREDICTION_DAYS = 5
N_ESTIMATORS = 200
MAX_DEPTH = 6
LEARNING_RATE = 0.1


# 自定义归一化类
class MinMaxScaler:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, X):
        self.min_val = np.min(X, axis=0)
        self.max_val = np.max(X, axis=0)

    def transform(self, X):
        return (X - self.min_val) / (self.max_val - self.min_val + 1e-8)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * (self.max_val - self.min_val) + self.min_val


# 自定义评估指标
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# 获取股票日线数据
def load_stock_data(stock_code, start_date, end_date):
    try:
        # 转换日期格式
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        # 使用data_fetcher模块中的函数
        df = fetch_stock_data(stock_code, start_str, end_str)

        if df is None or df.empty:
            print(f"未获取到{stock_code}的数据")
            return pd.DataFrame()

        # 处理数据
        df = df.sort_values('trade_date')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        df.index.name = 'Date'

        # 重命名列以符合后续处理
        # 注意：确保列名与数据中的列匹配
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'vol': 'Volume'
        }
        # 只重命名存在的列
        for orig, new in column_mapping.items():
            if orig in df.columns:
                df.rename(columns={orig: new}, inplace=True)

        # 确保必须有Close列
        if 'Close' not in df.columns:
            # 尝试使用小写close
            if 'close' in df.columns:
                df['Close'] = df['close']
            else:
                raise KeyError("数据中没有'Close'或'close'列，请检查数据格式！")

        # 保证Close列为float
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        # 如果有Volume列，也转换为数值
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"处理股票数据时出错: {str(e)}")
        return pd.DataFrame()


# 特征工程
def create_features(df):
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

    # 处理除零错误
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


# 预测函数 - 使用固定参数
def predict_stock_price(data, prediction_days=5, stock_code=None):
    try:
        # 确保数据不为空
        if data.empty:
            print("输入数据为空，无法预测")
            return pd.DataFrame(), (0, 0), (0, 0), None, None

        # 准备数据
        df = create_features(data)
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

        # 转换为DMatrix格式 - 使用XGBoost原生API
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train.values)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test.values)

        # 设置固定参数
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

        # 保存评估结果
        if stock_code is not None:
            eval_df = pd.DataFrame({
                'set_type': ['train'] * len(y_train) + ['test'] * len(y_test),
                'true': np.concatenate([y_train.values, y_test.values]),
                'pred': np.concatenate([train_pred, test_pred]),
                'residual': np.concatenate([y_train.values - train_pred, y_test.values - test_pred])
            })
            eval_df.to_csv(f'{stock_code}_XGBoost评估.csv', index=False)
            # 另存RMSE/MAE
            with open(f'{stock_code}_XGBoost评估.txt', 'w', encoding='utf-8') as f:
                f.write(f"训练集RMSE: {np.sqrt(train_mse):.4f}, MAE: {train_mae:.4f}\n")
                f.write(f"测试集RMSE: {np.sqrt(test_mse):.4f}, MAE: {test_mae:.4f}\n")

        # 预测未来
        future_dates = pd.date_range(start=df.index[-1] + datetime.timedelta(days=1), periods=prediction_days)
        predictions = []
        feature_columns = features.columns.tolist()  # 保证特征顺序和数量一致
        current_data = df.iloc[-1].copy()

        for i in range(prediction_days):
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


# 绘制股价图表
def plot_stock_data(data, prediction, stock_name):
    plt.figure(figsize=(12, 6))

    # 历史数据
    plt.plot(data.index, data['Close'], 'b-', label='Historical Close Price', linewidth=2)

    # 预测数据
    if isinstance(prediction, pd.DataFrame):
    if not prediction.empty:
            plt.plot(prediction.index, prediction['Predicted'], 'ro-', label='Predicted Price', linewidth=2)
            # 添加预测起始点
            plt.plot(data.index[-1], data['Close'].iloc[-1], 'go', markersize=8, label='Prediction Start')
    elif isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
        # 处理list或array类型的预测值
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(prediction))
        plt.plot(future_dates, prediction, 'ro-', label='Predicted Price', linewidth=2)
        # 添加预测起始点
        plt.plot(data.index[-1], data['Close'].iloc[-1], 'go', markersize=8, label='Prediction Start')
    else:
        # 预测为空的情况
        plt.plot(data.index[-1], data['Close'].iloc[-1], 'go', markersize=8, label='Prediction Start')

    # 添加移动平均线
    if 'MA5' in data:
        plt.plot(data.index, data['MA5'], 'y-', label='MA5', alpha=0.7)
    if 'MA10' in data:
        plt.plot(data.index, data['MA10'], 'm-', label='MA10', alpha=0.7)
    if 'MA20' in data:
        plt.plot(data.index, data['MA20'], 'c-', label='MA20', alpha=0.7)

    # 设置图表格式
    plt.title(f"{stock_name} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (CNY)")
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


def predict_and_save(stock_code, start_date, end_date, prediction_days=5):
    stock_data = load_stock_data(stock_code, start_date, end_date)
    if stock_data.empty:
        print(f"股票 {stock_code} 数据为空，跳过")
        return
    stock_name = stock_code
    prediction, train_metrics, test_metrics, pred_list, scaler = predict_stock_price(stock_data, prediction_days, stock_code=stock_code)
    if pred_list is not None and len(pred_list) == prediction_days:
        # 保存XGBoost预测结果
        result_df = pd.DataFrame({
            'date': pd.date_range(start=stock_data.index[-1] + datetime.timedelta(days=1), periods=prediction_days),
            'predicted_price': pred_list,
            'model': 'XGBoost'
        })
        result_df.to_csv(f'{stock_code}_XGBoost预测结果.csv', index=False)
        
        # 绘制图表
        img_path = plot_stock_data(stock_data, pred_list, stock_name)
        
        print(f"预测完成！")
        print(f"XGBoost预测结果已保存到: {stock_code}_XGBoost预测结果.csv")
        print(f"预测图表已保存到: {img_path}")
    else:
        print(f"股票 {stock_code} 预测失败")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stock_code', nargs='?', type=str, help='股票代码')
    parser.add_argument('--stock_code', type=str, dest='stock_code_named', help='股票代码（命名参数）')
    parser.add_argument('--days', type=int, default=5, help='预测天数')
    args = parser.parse_args()
    
    # 优先使用位置参数，如果没有则使用命名参数
    stock_code = args.stock_code or args.stock_code_named
    
            end_date = datetime.date(2025, 6, 26)
    start_date = end_date - datetime.timedelta(days=180)
    
    if stock_code:
        print(f"只预测单只股票: {stock_code}")
        predict_and_save(stock_code, start_date, end_date, args.days)
    else:
        print("批量预测所有沪深300成分股...")
        components_df = fetch_hs300_components()
        if components_df is None or components_df.empty:
            print("无法获取沪深300成分股")
            return
        latest_date = components_df['trade_date'].max()
        components = components_df[components_df['trade_date'] == latest_date]['con_code'].unique().tolist()
        all_predictions = []
        for ts_code in components:
            print(f"正在处理股票: {ts_code}")
            stock_data = load_stock_data(ts_code, start_date, end_date)
            if stock_data.empty:
                print(f"股票 {ts_code} 数据为空，跳过")
                continue
            stock_name = ts_code
            prediction, train_metrics, test_metrics, pred_list, scaler = predict_stock_price(stock_data, args.days, stock_code=ts_code)
            if pred_list is not None and len(pred_list) == args.days:
                # 保存XGBoost预测结果到CSV文件
                result_df = pd.DataFrame({
                    'date': pd.date_range(start=stock_data.index[-1] + datetime.timedelta(days=1), periods=args.days),
                    'predicted_price': pred_list,
                    'model': 'XGBoost'
                })
                result_df.to_csv(f'{ts_code}_XGBoost预测结果.csv', index=False)
                
                all_predictions.append({
                    'ts_code': ts_code,
                    'pred_next_n': pred_list
                })
                img_path = plot_stock_data(stock_data, pred_list, stock_name)
                print(f"预测图表已保存至: {img_path}")
                print(f"XGBoost预测结果已保存到: {ts_code}_XGBoost预测结果.csv")
                print(f"训练集 MSE: {train_metrics[0]:.4f}, MAE: {train_metrics[1]:.4f}")
                print(f"测试集 MSE: {train_metrics[0]:.4f}, MAE: {test_metrics[1]:.4f}")
            else:
                print(f"股票 {ts_code} 预测失败")
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv(f'all_hsz300_predictions_{args.days}days.csv', index=False)
        print(f"✅ 所有沪深300成分股预测完成，结果已保存至 all_hsz300_predictions_{args.days}days.csv")


if __name__ == "__main__":
    main()