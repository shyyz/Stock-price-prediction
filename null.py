# ... existing code ...
        # 只保留最基础的数据量检查
        if len(data) < 30:
            print(f"{stock_code} 数据量太小({len(data)})，跳过")
            return [0.0] * steps
        # 常规特征工程
        data['MA5'] = data[target_col].rolling(window=5).mean()
        data['MA10'] = data[target_col].rolling(window=10).mean()
        data['MA20'] = data[target_col].rolling(window=20).mean()
        data['RSI'] = calculate_rsi(data[target_col], window=14)
        data['volatility'] = data[target_col].pct_change().rolling(window=20).std().fillna(0)
        data['EMA12'] = data[target_col].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data[target_col].ewm(span=26, adjust=False).mean()
        data['MACD'] = (data['EMA12'] - data['EMA26']).fillna(0)
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        # 滞后特征
        for i in range(1, 6):
            data[f'lag_{i}'] = data[target_col].shift(i)
        data = data.dropna()
# ... existing code ...