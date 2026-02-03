骨转移智能预测系统 (PyCaret AutoML 版)
这是一个基于 Streamlit 和 PyCaret 构建的医疗数据挖掘与疾病预测系统。该系统利用 AutoML（自动化机器学习）技术，能够自动对比 15+ 种机器学习算法，筛选出最优模型，并提供特征重要性分析和临床预测功能。
系统内置了 SQLite 数据库，支持项目管理、历史回溯和状态保存。
🌟 核心功能
项目化管理：
创建新实验项目，记录项目名称与描述。
内置 SQLite 数据库，自动保存训练好的模型、原始数据和筛选结果。
支持随时加载历史项目，继续之前的分析工作。
AutoML 自动建模 (Step 1)：
一键竞赛：集成 PyCaret 引擎，自动执行数据清洗、缺失值填充、标准化。
算法对比：自动训练并评估 XGBoost, LightGBM, Random Forest, Logistic Regression 等多种算法。
最优推荐：根据 AUC 指标自动锁定表现最好的模型（Pipeline）。
特征筛选 (Step 2)：
基于最优模型自动计算特征重要性。
可视化展示特征贡献度排名。
支持用户交互式筛选 Top-K 关键特征，并更新至数据库。
临床预测应用 (Step 3)：
根据筛选出的特征，动态生成数据录入表单。
调用封装好的 Pipeline 进行端到端预测（自动处理预处理步骤）。
输出风险概率（高风险/低风险）及置信度。
🛠️ 安装与环境配置
由于 PyCaret 依赖较多，建议使用 Anaconda 创建独立的虚拟环境。
1. 环境要求
Python: 建议版本 3.9 或 3.10 (PyCaret 对 Python 3.12+ 支持尚不稳定)。
OS: Windows, macOS, 或 Linux。
2. 创建虚拟环境 (推荐)
code
Bash
conda create -n bone_mets_env python=3.10
conda activate bone_mets_env
3. 安装依赖
code
Bash
pip install pycaret streamlit pandas numpy shap matplotlib xgboost
(注意：在 Windows 上如果遇到安装错误，可能需要安装 Microsoft C++ Build Tools)
🚀 快速启动
下载本项目代码为 app_bone_mets_pycaret.py。
在终端运行以下命令：
code
Bash
streamlit run app_bone_mets_pycaret.py
浏览器将自动打开系统界面（默认地址：http://localhost:8501）。
📖 使用指南
步骤一：AutoML 自动训练
在侧边栏选择 “🆕 新建项目”。
输入项目名称和描述。
数据来源：
生成数据：点击生成并下载带有医学逻辑的模拟数据（CSV）。
上传数据：上传您的 CSV 训练数据，并选择“目标列”（Target/Outcome）。
点击 “🔥 启动 AutoML 引擎”。系统将运行数分钟，完成后显示模型排行榜，并自动保存冠军模型。
步骤二：特征重要性分析
点击左侧导航栏的 步骤二。
设置需要保留的特征数量（Top K）。
点击 “提取关键特征”。
系统会分析模型，展示条形图，并将筛选出的关键特征保存到数据库。
步骤三：临床预测应用
点击左侧导航栏的 步骤三。
界面会根据步骤二筛选出的特征，自动生成对应的输入框。
输入患者指标，点击 “开始预测”。
系统将输出患病风险概率。
加载历史项目
在侧边栏的下拉菜单中，选择之前保存的项目（格式：#ID 项目名 (日期)）。
系统会从 SQLite 数据库中瞬间恢复所有数据、模型和状态，您可以直接跳转到步骤三进行预测。
📂 文件结构说明
app_bone_mets_pycaret.py: 应用程序主代码文件。
bone_mets_pycaret.db: (自动生成) SQLite 数据库文件，存储所有项目历史。
*.pkl: (运行时临时) PyCaret 可能会生成的临时管道文件。
⚠️ 注意事项
数据格式：上传的 CSV 文件必须包含表头，且尽量只包含数值型特征（虽然 PyCaret 能处理分类变量，但建议预先清洗）。
模型解释性：部分复杂的集成模型（如 Stacking 或特定版本的 LightGBM）可能不支持原生的 feature_importances_ 属性，系统内置了代理逻辑来处理这种情况。
性能：AutoML 过程较为耗时，根据数据量大小，训练时间可能在 10秒 到 数分钟 不等。
