#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import os
from typing import Optional

import torch
import torch.distributed as dist
from auto_argparse import parse_args_and_run
from torch.multiprocessing import spawn

from src.baseline.trainer import TrainerCtc



def ddp_train_wrapper(ddp_local_rank,
                      train_kwargs
                      ):
    "Distributed Data Parallel Training"
    # setup ddp env
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=ddp_local_rank,
                            world_size=train_kwargs['ddp_nodes_num'])
    torch.cuda.set_device(ddp_local_rank)
    train_kwargs['ddp_local_rank'] = ddp_local_rank
    trainer = TrainerCtc(**train_kwargs)
    trainer.train()
    # clear ddp env
    dist.destroy_process_group()


def train_entrance(in_model_dir: str = 'pretrained_model/chinese-roberta-wwm-ext',
                   out_model_dir: str = 'model/ctc',
                   epochs: int = 10,
                   batch_size: int = 64,
                   learning_rate: float = 5e-5,
                   max_seq_len: int = 128,
                   train_fp: str = 'data/example.txt',
                   dev_fp: str = None,
                   test_fp: str = None,
                   random_seed_num: int = 42,
                   check_val_every_n_epoch: Optional[float] = 0.5,
                   early_stop_times: Optional[int] = 100,
                   freeze_embedding: bool = False,
                   warmup_steps: int = -1,
                   max_grad_norm: Optional[float] = None,
                   dev_data_ratio: Optional[float] = 0.2,
                   with_train_epoch_metric: bool = False,
                   training_mode: str = 'normal',
                   amp: Optional[bool] = True):
    """_summary_

    Args:
        # in_model_dir 预训练模型目录
        # out_model_dir 输出模型目录
        # epochs 训练轮数
        # batch_size batch文本数
        # max_seq_len 最大句子长度
        # learning_rate 学习率
        # train_fp 训练集文件
        # test_fp 测试集文件
        # dev_data_ratio  没有验证集时，会从训练集按照比例分割出验证集
        # random_seed_num 随机种子
        # check_val_every_n_epoch 每几轮对验证集进行指标计算
        # training_mode 训练模式 包括 ddp，dp, normal，分别代表分布式，并行，普通训练
        # amp 是否开启混合精度
        # freeze_embedding 是否冻结bert embed层
    """

    signature = inspect.signature(train_entrance)
    train_kwargs = {}
    for param in signature.parameters.values():
        train_kwargs[param.name] = eval(param.name)

    if training_mode in ('normal', 'dp'):
        trainer = TrainerCtc(**train_kwargs)
        trainer.train()

    elif training_mode == 'ddp':
        ddp_nodes_num = torch.cuda.device_count()
        train_kwargs['ddp_nodes_num'] = ddp_nodes_num
        spawn(ddp_train_wrapper,
              args=(train_kwargs,),
              nprocs=ddp_nodes_num,
              join=True)


if __name__ == '__main__':
    parse_args_and_run(train_entrance)
