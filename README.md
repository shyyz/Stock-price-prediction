# Stock-price-prediction
# 安装与部署说明

## 一、准备工作
1. **注册Tushare并获取API接口**
   - 访问[Tushare官网](https://tushare.pro/register)，注册账号并登录。
   - 在个人中心中获取API Token。此Token用于访问Tushare提供的金融数据接口。
   - 如果你不想注册Tushare，也可以直接使用本代码中提供的默认Token（不推荐，因为可能会受到使用限制）。

2. **注册GitHub账号**
   - 如果你还没有GitHub账号，请访问[GitHub官网](https://github.com)并注册。
   - 登录后，确保你的账号信息完整，以便后续操作。

3. **注册Streamlit账号**
   - 访问[Streamlit官网](https://streamlit.io/cloud)并使用GitHub账号登录。
   - 完成注册流程后，你将获得一个Streamlit的个人空间，用于部署Web应用。

## 二、项目部署步骤
1. **从GitHub克隆项目**
   - 打开终端或命令行工具。
   - 输入以下命令克隆项目到本地：
     ```bash
     git clone https://github.com/shyyz/Stock-price-prediction.git
     ```
   - 进入项目目录：
     ```bash
     cd Stock-price-prediction
     ```

2. **检查项目依赖**
   - 确保你的本地环境中已安装Python（推荐Python 3.8及以上版本）。
   - 在项目目录中运行以下命令安装依赖：
     ```bash
     pip install -r requirements.txt
     ```
     如果`requirements.txt`文件不存在，请检查项目是否包含依赖文件，或者手动安装项目中提到的依赖库。

3. **在Streamlit上部署**
   - 登录Streamlit官网，进入你的个人空间。
   - 点击“Create New App”按钮，开始创建新的应用。
   - 在弹出的界面中，选择“GitHub”作为代码来源。
   - 输入你的GitHub仓库地址（例如：`https://github.com/shyyz/Stock-price-prediction`）。
   - 选择`main.py`作为入口文件。
   - 点击“Deploy”按钮，Streamlit将自动部署你的项目。
## 三、网站使用说明

1. **使用功能**
   - 在网站页面中，您将看到股票价格预测的相关功能和界面。
   - 输入所需的股票代码或选择相应的股票名称，网站将根据历史数据进行价格预测。
   - 查看预测结果和相关图表，帮助您更好地了解股票价格的走势。

2. **注意事项**
   - 请确保网络连接稳定，以便正常访问和使用网站功能。
   - 如果遇到任何问题，可以尝试刷新页面或稍后再试。
   - 请注意，股票价格预测仅供参考，实际投资决策应结合更多专业分析和个人判断。 
