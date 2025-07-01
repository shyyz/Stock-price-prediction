# app.py - 股票预测分析平台前端应用
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import random
import pandas as pd
import os
import datetime
import data_fetcher as data
import null
import xgboots as xgb
import traceback
import numpy as np

class StockPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("股票预测分析平台")
        self.root.geometry("1400x900")

        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 标题
        self.title_label = ttk.Label(
            self.main_frame,
            text="股票预测分析平台",
            font=("Arial", 20, "bold")
        )
        self.title_label.pack(pady=20)

        # 输入区域
        self.input_frame = ttk.LabelFrame(self.main_frame, text="股票选择")
        self.input_frame.pack(fill=tk.X, padx=10, pady=10)

        # 股票代码输入
        ttk.Label(self.input_frame, text="股票代码:").grid(row=0, column=0, padx=5, pady=5)
        self.stock_code_entry = ttk.Entry(self.input_frame, width=15)
        self.stock_code_entry.grid(row=0, column=1, padx=5, pady=5)
        self.stock_code_entry.insert(0, "600519.SH")

        # 获取数据按钮
        self.fetch_button = ttk.Button(
            self.input_frame,
            text="获取数据并预测",
            command=self.fetch_and_predict
        )
        self.fetch_button.grid(row=0, column=2, padx=10, pady=5)

        # 随机选择按钮
        self.random_button = ttk.Button(
            self.input_frame,
            text="随机选择股票",
            command=self.select_random_stock
        )
        self.random_button.grid(row=0, column=3, padx=10, pady=5)

        # 结果展示区域
        self.result_frame = ttk.Frame(self.main_frame)
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        # 随机森林结果
        self.rf_frame = ttk.LabelFrame(self.result_frame, text="随机森林预测")
        self.rf_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.rf_container = ttk.Frame(self.rf_frame)
        self.rf_container.pack(fill=tk.BOTH, expand=True)

        # XGBoost结果
        self.xgb_frame = ttk.LabelFrame(self.result_frame, text="XGBoost预测")
        self.xgb_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.xgb_container = ttk.Frame(self.xgb_frame)
        self.xgb_container.pack(fill=tk.BOTH, expand=True)

        # 配置网格权重
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.columnconfigure(1, weight=1)
        self.result_frame.rowconfigure(0, weight=1)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(
            root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def init_result_components(self):
        for widget in self.rf_container.winfo_children():
            widget.destroy()
        for widget in self.xgb_container.winfo_children():
            widget.destroy()
        self.rf_canvas = tk.Canvas(self.rf_container, width=400, height=300)
        self.rf_canvas.pack(fill=tk.BOTH, expand=True)
        self.rf_prediction_frame = ttk.Frame(self.rf_container)
        self.rf_prediction_frame.pack(fill=tk.X, pady=10)
        self.xgb_canvas = tk.Canvas(self.xgb_container, width=400, height=300)
        self.xgb_canvas.pack(fill=tk.BOTH, expand=True)
        self.xgb_prediction_frame = ttk.Frame(self.xgb_container)
        self.xgb_prediction_frame.pack(fill=tk.X, pady=10)

    def select_random_stock(self):
        if hasattr(data, 'get_hs300_stocks'):
            stocks = data.get_hs300_stocks()
        else:
            # 兼容性处理
            df = data.fetch_hs300_components()
            if df is not None and not df.empty and 'con_code' in df.columns:
                stocks = [(row['con_code'], row['name'] if 'name' in row else row['con_code']) for _, row in df.iterrows()]
            else:
                stocks = []
        if not stocks:
            messagebox.showerror("错误", "无法获取沪深300成分股")
            return
        stock_code, stock_name = random.choice(stocks)
        self.stock_code_entry.delete(0, tk.END)
        self.stock_code_entry.insert(0, stock_code)
        self.fetch_and_predict()

    def fetch_and_predict(self):
        stock_code = self.stock_code_entry.get().strip()
        if not stock_code:
            messagebox.showwarning("警告", "请输入股票代码")
            return
        stock_name = self.get_stock_name(stock_code)
        self.init_result_components()
        loading_label = ttk.Label(self.result_frame, text=f"正在获取{stock_code}的数据并预测...", font=("Arial", 12))
        loading_label.grid(row=1, column=0, columnspan=2, pady=20)
        self.root.update()
        try:
            end_date = datetime.date(2025, 6, 26)
            start_date = end_date - datetime.timedelta(days=180)
            self.status_var.set(f"正在获取{stock_code}的数据...")
            self.root.update()
            stock_data = data.fetch_stock_data(stock_code, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
            if stock_data is None or stock_data.empty:
                self.status_var.set("数据获取失败")
                messagebox.showerror("错误", f"无法获取{stock_code}的股票数据")
                loading_label.destroy()
                return
            # 统一索引和列名，兼容null和xgboots
            if 'trade_date' in stock_data.columns:
                stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
                stock_data = stock_data.sort_values('trade_date')
                stock_data = stock_data.set_index('trade_date')
                stock_data = stock_data.rename(columns={'trade_date': 'date'})
            # 1. 给null.py用的小写列名
            stock_data_for_null = stock_data.copy()
            # 2. 给xgboots.py用的大写列名
            rename_dict = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'vol': 'Volume'
            }
            stock_data_for_xgb = stock_data.rename(columns={k: v for k, v in rename_dict.items() if k in stock_data.columns})
            # 确保价格列为float类型
            for col in ['close', 'Close', 'open', 'Open', 'high', 'High', 'low', 'Low']:
                if col in stock_data_for_null.columns:
                    stock_data_for_null[col] = pd.to_numeric(stock_data_for_null[col], errors='coerce')
                if col in stock_data_for_xgb.columns:
                    stock_data_for_xgb[col] = pd.to_numeric(stock_data_for_xgb[col], errors='coerce')
            # 随机森林预测
            self.status_var.set(f"正在使用随机森林预测{stock_code}...")
            self.root.update()
            rf_pred = null.create_multi_step_target(stock_data_for_null, steps=5)
            rf_img = null.plot_kline(stock_code, stock_data_for_null)
            # XGBoost预测
            self.status_var.set(f"正在使用XGBoost预测{stock_code}...")
            self.root.update()
            try:
                print("开始XGBoost预测...")
                xgb_result, xgb_metrics, xgb_mae_metrics, xgb_predictions, xgb_scaler = xgb.predict_stock_price(stock_data_for_xgb)
                print(f"XGBoost预测完成，结果类型: {type(xgb_predictions)}")
                if xgb_predictions is not None:
                    print(f"XGBoost预测值数量: {len(xgb_predictions)}")
                    print(f"XGBoost预测值: {xgb_predictions[:3]}...")  # 显示前3个预测值
                else:
                    print("XGBoost预测返回None")
            except Exception as e:
                print(f"XGBoost预测出错: {str(e)}")
                import traceback
                traceback.print_exc()
                xgb_result = pd.DataFrame()
                xgb_metrics = (0, 0)
                xgb_mae_metrics = (0, 0)
                xgb_predictions = None
                xgb_scaler = None
            xgb_img = xgb.plot_stock_data(stock_data_for_xgb, xgb_predictions, stock_name)
            loading_label.destroy()
            self.display_results(
                stock_code, stock_name,
                rf_pred, rf_img,
                xgb_predictions, xgb_metrics, xgb_mae_metrics, xgb_img
            )
            self.status_var.set(f"完成{stock_code}的预测分析")
        except Exception as e:
            if 'loading_label' in locals():
                loading_label.destroy()
            messagebox.showerror("错误", f"预测过程中出错: {str(e)}")
            traceback.print_exc()
            self.status_var.set("预测失败")

    def get_stock_name(self, stock_code):
        try:
            stocks_df = data.fetch_hs300_components()
            if not stocks_df.empty and 'con_code' in stocks_df.columns and 'name' in stocks_df.columns:
                match = stocks_df[stocks_df['con_code'] == stock_code]
                if not match.empty:
                    return match.iloc[0]['name']
            return stock_code
        except Exception as e:
            print(f"获取股票名称失败: {str(e)}")
            return stock_code

    def display_results(self, stock_code, stock_name, rf_pred, rf_img, xgb_pred, xgb_metrics, xgb_mae_metrics, xgb_img):
        # 随机森林K线图
        print("加载K线图路径：", rf_img, "文件是否存在：", os.path.exists(rf_img) if rf_img else "路径为None")
        if rf_img and os.path.exists(rf_img):
            img = Image.open(rf_img)
            img = img.resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.rf_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.rf_canvas.image = photo
        else:
            self.rf_canvas.create_text(100, 100, text="随机森林K线图未找到", fill="red")
        self.display_predictions(self.rf_prediction_frame, rf_pred, None, None, None, "随机森林")
        # XGBoost K线图
        print("加载XGBoost图路径：", xgb_img, "文件是否存在：", os.path.exists(xgb_img) if xgb_img else "路径为None")
        if xgb_img and os.path.exists(xgb_img):
            img = Image.open(xgb_img)
            img = img.resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.xgb_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.xgb_canvas.image = photo
        else:
            self.xgb_canvas.create_text(100, 100, text="XGBoost K线图未找到", fill="red")
        self.display_predictions(self.xgb_prediction_frame, xgb_pred, None, xgb_metrics[0] if xgb_metrics else None, xgb_mae_metrics[1] if xgb_mae_metrics else None, "XGBoost")

    def display_predictions(self, frame, predictions, changes, mse, mae, model_name):
        for widget in frame.winfo_children():
            widget.destroy()
        if predictions is None or len(predictions) == 0:
            if model_name == "XGBoost":
                ttk.Label(frame, text="XGBoost预测失败，请检查数据", foreground="red").pack(pady=10)
            else:
                ttk.Label(frame, text=f"{model_name}预测失败", foreground="red").pack(pady=10)
            return
        model_label = ttk.Label(frame, text=f"{model_name}预测结果:", font=("Arial", 10, "bold"))
        model_label.pack(anchor=tk.W, pady=5)
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(table_frame, text="天数", width=8, anchor=tk.CENTER).grid(row=0, column=0)
        ttk.Label(table_frame, text="预测价格", width=10, anchor=tk.CENTER).grid(row=0, column=1)
        for i in range(len(predictions)):
            ttk.Label(table_frame, text=f"第{i + 1}天", width=8).grid(row=i + 1, column=0)
            # 处理不同类型的预测值
            val = predictions[i]
            if isinstance(val, (pd.Series, pd.DataFrame, np.ndarray)):
                val = float(val.values.flatten()[0])
            elif isinstance(val, (list, tuple)):
                val = float(val[0])
            elif not isinstance(val, (int, float)):
                val = float(val)
            price_label = ttk.Label(table_frame, text=f"{val:.2f}", width=10)
            price_label.grid(row=i + 1, column=1)
        if model_name == "XGBoost" and mse is not None and mae is not None:
            metrics_frame = ttk.Frame(frame)
            metrics_frame.pack(fill=tk.X, padx=10, pady=10)
            ttk.Label(metrics_frame, text="模型评估:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W)
            ttk.Label(metrics_frame, text=f"MSE: {mse:.4f}").grid(row=1, column=0, sticky=tk.W, padx=5)
            ttk.Label(metrics_frame, text=f"MAE: {mae:.4f}").grid(row=2, column=0, sticky=tk.W, padx=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictorApp(root)
    root.mainloop()