# bugs_translater

## 项目概述
本项目是一个基于RAG技术的翻译系统，通过结合语法规则，词典和平行语料作为参考，构建高质量的prompt，利用大模型实现较为准确，符合语言习惯的翻译。
### 联系小熊虫(bearischen@gmail.com/2300013202@stu.pku.edu.cn)
![img](https://github.com/miakasaever/llm_debate/blob/main/%E7%86%8A%E8%99%AB.jpg)

## 功能特点

1. **多信息检索**：从多个信息源中进行检索最相关信息来构建高质量prompt(我晒干了沉默vav无端联想)
2. **语义词频混合特征抽取**:使用预训练模型对待句子抽取语义特征并于tf-idf(或者bm25，可自选)特征进行混合来判断相似度
## 文件结构
```text
├── fc.py                 # 核心功能类实现
├── main.py               # 主程序入口
├── utils.py              # 工具函数
├── file_to_use/          # 数据文件目录
│   ├── grammar_book.json     # 语法规则
│   ├── dictionary_za2zh.jsonl  # 词典
│   ├── parallel_corpus.json   # 平行语料库
│   └── test_data_hard.json    # 测试数据
│   └── test_data_simple.json    # 测试数据（简单版）
├── results/              # 结果输出目录
│   ├── sample_submission.csv  # 提交文件模板
│   └── translation_prompt.json # 详细翻译结果

```
## 安装指南

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/bugs_translater.git
cd bugs_transolater 
```
2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 设置环境变量（获取Dashscope API密钥）
```bash
export DASHSCOPE_API_KEY=your_api_key_here
```
## 使用说明
### 基本运行
```bash
python main.py 
```

## 配置说明
1. 测试样例
此处提供两个测试样例``test_data_hard.json``以及``test_data_simple.json`` 。程序默认使用``test_data_hard.json``作为测试样例。
2. 使用配置
在``fc.py``可配置以下参数
```python
USE_SBERT = True  # 是否使用Sentence Transformers语义模型
RETRIEVAL_METHOD = 'BM25'  # 检索方法（可选'BM25'或'TFIDF'）
```
3. Acquire设置
第一次运行程序需要设置``acquire``中提及的各项参数来修改对于API的访问配置，在之后的运行中会直接启用配置好的json文件。
## 贡献指南
### 本项目还有以下可改进的地方
1. 暂时只支持壮语->汉语
2. 提供更好的语义抽取模型
3. 提供更好的检索方式
4. 提供更好的信息源
5. 配置更好的混合参数
6. 暂时只支持使用openai方式调用API
### 欢迎贡献，请遵守以下流程
1. Fork本项目 and and and and and 联系我(bearischen@gmail.com or 2300013202@stu.pku.edu.cn)
2. 创建新分支``git checkout -b feature/your-feature``
3. 提交更改``git commit -am 'Add some feature'``
4. 推送分支``git push origin feature/your-feature``
5. 创建pull request
