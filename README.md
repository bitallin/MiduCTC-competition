# 比赛

文本智能校对大赛baseline

## 代码结构
```
├── command
│   └── train.sh       # 训练脚本
├── data
├── logs
├── pretrained_model
└── src
    ├── __init__.py
    ├── baseline       # baseline系统
    ├── corrector.py   # 文本校对入口
    ├── evaluate.py    # 指标评估
    ├── metric.py      # 指标计算文件 
    ├── prepare_for_upload.py  # 生成要提交的结果文件
    └── train.py       # 训练入口
```

## 使用说明

- 数据集获取：请于比赛官网报名获取数据集
- 提供了基础校对系统的baseline，其中baseline模型训练参数说明参考src/baseline/trainer.py
- baseline中的预训练模型支持使用bert类模型，可从HuggingFace下载bert类预训练模型，如: [chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)等
- baseline上仅作参考，参赛队伍可对baseline进行二次开发，或采取其他解决方案。

## 开始训练

```
cd command && sh train.sh
```

## 其他公开数据集

- CGED历年公开数据集：http://www.cged.tech/
- NLPCC2018语法纠错数据集：http://tcci.ccf.org.cn/conference/2018/taskdata.php
- SIGHAN及相关训练集：http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html

