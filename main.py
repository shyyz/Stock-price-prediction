import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置页面配置 - 使用宽屏布局
st.set_page_config(
    page_title="股票价格预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置 Tushare Token
ts.set_token('4c16742e8a98b00a7308d158e35e43a05eb0b5d2f8874bcf9549520b')
pro = ts.pro_api()


def show_prediction_results(stock_code):
    """显示后端预测结果"""
    st.info("正在读取预测结果...")

    # XGBoost结果
    xgb_csv_path = f"stock_data/{stock_code}_XGBoost预测结果.csv"
    xgb_img_path = f"stock_data/{stock_code}_预测图表.png"

    # 随机森林结果
    rf_csv_path = f"stock_data/{stock_code}_随机森林预测结果.csv"
    rf_img_path = f"kline_plots/{stock_code}.png"

    # 展示XGBoost
    if os.path.exists(xgb_csv_path) and os.path.exists(xgb_img_path):
        st.success("XGBoost预测成功！")
        xgb_df = pd.read_csv(xgb_csv_path)
        st.dataframe(xgb_df)
        st.info(f"XGBoost预测文件内容：{xgb_csv_path}")
        st.info(f"XGBoost第一行预测价格：{xgb_df['predicted_price'].iloc[0] if len(xgb_df) > 0 else 'N/A'}")
        st.image(xgb_img_path, caption="XGBoost K线图", use_container_width=True)
    else:
        st.warning(f"XGBoost预测文件未找到：{xgb_csv_path} 或 {xgb_img_path}")

    # 展示随机森林
    if os.path.exists(rf_csv_path) and os.path.exists(rf_img_path):
        st.success("随机森林预测成功！")
        rf_df = pd.read_csv(rf_csv_path)
        st.dataframe(rf_df)
        st.info(f"随机森林预测文件内容：{rf_csv_path}")
        st.info(f"随机森林第一行预测价格：{rf_df['predicted_price'].iloc[0] if len(rf_df) > 0 else 'N/A'}")
        st.image(rf_img_path, caption="随机森林K线图", use_container_width=True)
    else:
        st.warning(f"随机森林预测文件未找到：{rf_csv_path} 或 {rf_img_path}")


def prediction_page():
    """预测页面"""
    st.title("📈 股票价格预测系统")
    st.markdown("---")

    # 使用容器来改善布局
    with st.container():
        # 股票代码输入区域
        st.subheader("🔍 股票选择")

        # 创建输入区域
        input_col1, input_col2, input_col3 = st.columns([3, 1, 1])

        with input_col1:
            stock_code = st.text_input(
                "请输入股票代码",
                placeholder="例如：000001.SZ",
                help="请输入完整的股票代码，包括后缀（.SZ 或 .SH）"
            )

        with input_col2:
            predict_button = st.button("🚀 开始预测", type="primary", use_container_width=True)

        with input_col3:
            query_button = st.button("🔍 查询股票", use_container_width=True)

        # 处理查询股票按钮点击
        if query_button:
            st.subheader("📋 沪深300成分股列表")
            try:
                # 读取沪深300成分股文件
                if os.path.exists('stock_data/hs300_components.csv'):
                    components_df = pd.read_csv('stock_data/hs300_components.csv')
                    # 显示完整的300个成分股
                    st.dataframe(components_df, use_container_width=True)
                    st.info(f"共 {len(components_df)} 只成分股，按权重排序。")
                else:
                    st.warning("未找到沪深300成分股文件，请确保 stock_data/hs300_components.csv 文件存在。")
            except Exception as e:
                st.error(f"读取成分股文件失败: {e}")

        # 处理预测按钮点击
        if predict_button:
            if stock_code:
                # 验证股票代码格式
                if not (stock_code.endswith('.SZ') or stock_code.endswith('.SH')):
                    st.error("❌ 请输入正确的股票代码格式（如：000001.SZ）")
                    return
                # 关键：保存股票代码到session_state
                st.session_state['stock_code'] = stock_code
                # 直接显示预测结果
                show_prediction_results(stock_code)
            else:
                st.warning("⚠️ 请输入股票代码")


def about_page():
    """使用说明页面"""
    st.title("使用说明")
    st.markdown("---")

    st.markdown("""
    ## 🎯 系统功能

    本系统是一个基于机器学习的股票价格预测工具，使用随机森林和XGBoost算法对未来5个交易日进行预测。

    ## 📋 使用步骤

    1. **输入股票代码**
       - 在预测页面输入完整的股票代码
       - 格式：股票代码.交易所后缀（如：000001.SZ）
       - 支持的交易所：SZ（深圳）、SH（上海）

    2. **开始预测**
       - 点击"开始预测"按钮
       - 系统会自动获取历史数据并运行预测模型
       - 预测过程可能需要几分钟时间

    3. **查看结果**
       - 随机森林预测结果表格
       - XGBoost预测结果表格
       - 预测K线图
       - 预测日期为未来5个交易日

    ## 🔧 技术说明

    - **数据源**：使用Tushare API获取股票历史数据
    - **特征工程**：包含移动平均线、RSI指标、滞后特征等
    - **预测模型**：随机森林和XGBoost两种算法
    - **预测周期**：未来5个交易日

    ## ⚠️ 注意事项

    - 预测结果仅供参考，不构成投资建议
    - 股票投资存在风险，请谨慎决策
    - 系统预测基于历史数据，市场变化可能影响预测准确性
    - 预测过程可能需要几分钟时间，请耐心等待

    ## 📞 技术支持

    如有问题，请联系技术支持团队。
    """)


def evaluation_page():
    """模型评估页面：双模型展示"""
    st.title("模型评估")
    st.markdown("---")
    stock_code = st.session_state.get('stock_code', None)
    if not stock_code:
        st.info("请先在预测页面输入股票代码并预测")
        return
    xgb_eval_path = f"stock_data/{stock_code}_XGBoost评估.csv"
    rf_eval_path = f"stock_data/{stock_code}_随机森林评估.csv"
    col1, col2 = st.columns(2)
    # XGBoost评估
    with col1:
        if os.path.exists(xgb_eval_path):
            xgb_eval = pd.read_csv(xgb_eval_path)
            train_rmse = xgb_eval[xgb_eval['set_type'] == 'train']['residual'].pow(2).mean() ** 0.5
            test_rmse = xgb_eval[xgb_eval['set_type'] == 'test']['residual'].pow(2).mean() ** 0.5
            # 新增MSE和R2
            train_mse = xgb_eval[xgb_eval['set_type'] == 'train']['mse'].iloc[0] if 'mse' in xgb_eval.columns else None
            test_mse = xgb_eval[xgb_eval['set_type'] == 'test']['mse'].iloc[0] if 'mse' in xgb_eval.columns else None
            train_r2 = xgb_eval[xgb_eval['set_type'] == 'train']['r2'].iloc[0] if 'r2' in xgb_eval.columns else None
            test_r2 = xgb_eval[xgb_eval['set_type'] == 'test']['r2'].iloc[0] if 'r2' in xgb_eval.columns else None
            st.metric("训练集RMSE", f"{train_rmse:.4f}")
            st.metric("测试集RMSE", f"{test_rmse:.4f}")
            if train_mse is not None and test_mse is not None:
                st.metric("训练集MSE", f"{train_mse:.4f}")
                st.metric("测试集MSE", f"{test_mse:.4f}")
            if train_r2 is not None and test_r2 is not None:
                st.metric("训练集R²", f"{train_r2:.4f}")
                st.metric("测试集R²", f"{test_r2:.4f}")
            st.markdown("**残差分布图**")
            st.line_chart(xgb_eval['residual'])
            st.markdown("**残差直方图**")
            st.bar_chart(xgb_eval['residual'])
        else:
            st.warning(f"未找到XGBoost评估文件: {xgb_eval_path}")

    # 随机森林评估
    with col2:
        if os.path.exists(rf_eval_path):
            rf_eval = pd.read_csv(rf_eval_path)
            train_rmse = rf_eval[rf_eval['set_type'] == 'train']['residual'].pow(2).mean() ** 0.5
            test_rmse = rf_eval[rf_eval['set_type'] == 'test']['residual'].pow(2).mean() ** 0.5
            # 新增MSE和R2
            train_mse = rf_eval[rf_eval['set_type'] == 'train']['mse'].iloc[0] if 'mse' in rf_eval.columns else None
            test_mse = rf_eval[rf_eval['set_type'] == 'test']['mse'].iloc[0] if 'mse' in rf_eval.columns else None
            train_r2 = rf_eval[rf_eval['set_type'] == 'train']['r2'].iloc[0] if 'r2' in rf_eval.columns else None
            test_r2 = rf_eval[rf_eval['set_type'] == 'test']['r2'].iloc[0] if 'r2' in rf_eval.columns else None
            st.metric("训练集RMSE", f"{train_rmse:.4f}")
            st.metric("测试集RMSE", f"{test_rmse:.4f}")
            if train_mse is not None and test_mse is not None:
                st.metric("训练集MSE", f"{train_mse:.4f}")
                st.metric("测试集MSE", f"{test_mse:.4f}")
            if train_r2 is not None and test_r2 is not None:
                st.metric("训练集R²", f"{train_r2:.4f}")
                st.metric("测试集R²", f"{test_r2:.4f}")
            st.markdown("**残差分布图**")
            st.line_chart(rf_eval['residual'])
            st.markdown("**残差直方图**")
            st.bar_chart(rf_eval['residual'])
        else:
            st.warning(f"未找到随机森林评估文件: {rf_eval_path}")


def main():
    """主函数"""
    # 侧边栏导航
    st.sidebar.title("📊 股票预测系统")
    st.sidebar.markdown("---")

    # 直接显示页面选项
    st.sidebar.markdown("### 📄 页面导航")

    if st.sidebar.button("📈 股票预测", use_container_width=True):
        st.session_state.page = "股票预测"

    if st.sidebar.button("📖 使用说明", use_container_width=True):
        st.session_state.page = "使用说明"

    if st.sidebar.button("📊 模型评估", use_container_width=True):
        st.session_state.page = "模型评估"

    # 初始化页面状态
    if 'page' not in st.session_state:
        st.session_state.page = "股票预测"

    # 根据选择的页面显示内容
    if st.session_state.page == "股票预测":
        prediction_page()
    elif st.session_state.page == "使用说明":
        about_page()
    elif st.session_state.page == "模型评估":
        evaluation_page()


if __name__ == "__main__":
    main()