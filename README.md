# 文本智能校对大赛


- [文本智能校对大赛](#文本智能校对大赛)
  - [日历](#日历)
  - [任务描述](#任务描述)
  - [Baseline介绍](#baseline介绍)
    - [模型](#模型)
    - [代码结构](#代码结构)
    - [使用说明](#使用说明)
    - [开始训练](#开始训练)
  - [其他公开数据集](#其他公开数据集)
  - [相关资源](#相关资源)
 

## 日历


| 时间 | 事件 |
| ------- | ------- |
|  2022.7.13 | 比赛启动，开放报名，[赛事网址](https://aistudio.baidu.com/aistudio/competition/detail/404/0/introduction)，初赛A榜数据集，初赛A榜提交入口|
| 2022.8.12 | 报名截止，关闭初赛A榜评测入口 |
| 2022.8.13 | 开放初赛B榜数据集、评测入口 |
| 2022.8.17 | 关闭初赛B榜数据集、评测入口 |
| 2022.8.18 | 开放决赛数据集、评测入口 |
| 2022.8.20 | 关闭决赛数据集、评测入口 |



## 任务描述

本次赛题选择网络文本作为输入，从中检测并纠正错误，实现中文文本校对系统。即给定一段文本，校对系统从中检测出错误字词、错误类型，并进行纠正，最终输出校正后的结果。

文本校对又称文本纠错，相关资料可参考自然语言处理方向的**语法纠错（Grammatical Error Correction, GEC)** 任务和**中文拼写纠错（Chinese spelling check, CSC）**和 [一些开放资料](#相关资源)



## Baseline介绍

### 模型

提供了**GECToR**作为baseline模型，可参考[GECToR论文](https://aclanthology.org/2020.bea-1.16.pdf)和[GECToR源代码](https://github.com/grammarly/gector)



### 代码结构
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

### 使用说明

- 数据集获取：请于比赛官网报名获取数据集
- 提供了基础校对系统的baseline，其中baseline模型训练参数说明参考src/baseline/trainer.py
- baseline中的预训练模型支持使用bert类模型，可从HuggingFace下载bert类预训练模型，如: [chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)等
- baseline仅作参考，参赛队伍可对baseline进行二次开发，或采取其他解决方案。

### 开始训练

```
cd command && sh train.sh
```

## 其他公开数据集

- CGED历年公开数据集：http://www.cged.tech/
- NLPCC2018语法纠错数据集：http://tcci.ccf.org.cn/conference/2018/taskdata.php
- SIGHAN及相关训练集：http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html

## 相关资源

- [pycorrector](https://github.com/shibing624/pycorrector)
- [中文文本纠错开源项目整理](https://github.com/li-aolong/li-aolong.github.io/issues/12)
- [PyCorrector文本纠错工具实践和代码详解](https://zhuanlan.zhihu.com/p/138981644)
- [CTC-2021](https://github.com/destwang/CTC2021)
- [Text Correction Papers](https://github.com/nghuyong/text-correction-papers)
- [文本语法纠错不完全调研：学术界 v.s. 工业界最新研究进展](https://zhuanlan.zhihu.com/p/398928434)
- [知物由学 | “找茬”不如交给AI算法，细说文本纠错的多种实现途径 ](https://zhuanlan.zhihu.com/p/434672168)
- [中文文本纠错算法--错别字纠正的二三事  ](https://zhuanlan.zhihu.com/p/40806718)
- [中文文本纠错算法走到多远了？](https://cloud.tencent.com/developer/article/1435917)
- [平安寿险 AI 团队 | 文本纠错技术探索和实践](https://www.6aiq.com/article/1594474039153)
- [中文文本纠错（Chinese Text Correction, CTC）相关资源，本资源由哈工大讯飞联合实验室（HFL）王宝鑫和赵红红整理维护。](https://github.com/destwang/CTCResources)