#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from math import ceil
from typing import List
import json
import numpy as np
import torch
from rich.progress import track
from src import logger
from src.baseline.dataset import DatasetCTC
from src.baseline.modeling import ModelingCtcBert
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AdamW
from src.baseline.tokenizer import CtcTokenizer
from typing import Optional
from transformers.optimization import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm
from src.metric import ctc_f1


class TrainerCtc:
    def __init__(self,
                 in_model_dir: str,
                 out_model_dir: str,
                 epochs: int,
                 batch_size: int,
                 learning_rate: float,
                 max_seq_len: int,
                 train_fp: str,
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
                 loss_ignore_id = -100,
                 ctc_label_vocab_dir: str = 'src/baseline/ctc_vocab',
                 amp: Optional[bool] = True,
                 ddp_nodes_num: Optional[int] = 1,
                 ddp_local_rank: Optional[int] = -1,
                 **kwargs
                 ):
        
        """
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
        # warmup_steps 预热steps
        # check_val_every_n_epoch 每几轮对验证集进行指标计算
        # training_mode 训练模式 包括 ddp，dp, normal，分别代表分布式，并行，普通训练
        # amp 是否开启混合精度
        # freeze_embedding 是否冻结bert embed层
        """

        current_time = time.strftime("_%YY%mM%dD%HH", time.localtime())
        self.in_model_dir = in_model_dir
        self.out_model_dir = os.path.join(out_model_dir, '')[
            :-1] + current_time

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_len = max_seq_len
        self.random_seed_num = random_seed_num
        self.freeze_embedding = freeze_embedding
        self.train_fp = train_fp
        self.dev_fp = dev_fp
        self.test_fp = test_fp
        self.ctc_label_vocab_dir = ctc_label_vocab_dir
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.early_stop_times = early_stop_times
        self.dev_data_ratio = dev_data_ratio
        self._loss_ignore_id = loss_ignore_id
        assert training_mode in ('normal', 'dp', 'ddp')  # 普通，数据并行，分布式训练
        self.training_mode = training_mode
        self.ddp_nodes_num = ddp_nodes_num
        self.ddp_local_rank = int(ddp_local_rank)
        self.dev_data_ratio = dev_data_ratio
        self.amp = amp
        self._warmup_steps = warmup_steps
        self._max_grad_norm = max_grad_norm
        self.with_train_epoch_metric = with_train_epoch_metric
        if not os.path.exists(self.out_model_dir) and self.ddp_local_rank in (-1, 0):
            os.mkdir(self.out_model_dir)

        if self.amp:
            self.scaler = GradScaler()  # auto mixed precision
        self.fit_seed(self.random_seed_num)
        self.tokenizer = CtcTokenizer.from_pretrained(
            self.in_model_dir)
        self.train_ds, self.dev_ds, self.test_ds = self.load_data()
        self.model, self.optimizer, self.scheduler = self.load_suite()

        self._id2dtag, self._dtag2id, self._id2ctag, self._ctag2id = self.load_label_vocab()

        self._keep_id_in_ctag = self._ctag2id['$KEEP']

    @staticmethod
    def load_texts_from_fp(file_path):
        trg_texts, src_texts = [], []
        
        if '.txt' in file_path:
            for line in open(file_path, 'r', encoding='utf-8'):
                line = line.strip().split('\t')
                if line:
                    # 需注意txt文件中src和trg前后关系
                    src_texts.append(line[0])
                    trg_texts.append(line[1])
        elif '.json' in file_path:
            json_data = json.load(open(file_path, 'r', encoding='utf-8'))    
            for line in json_data:
                src_texts.append(line['source'])
                trg_texts.append(line['target'])
                
              
        return src_texts, trg_texts

    def load_label_vocab(self):
        dtag_fp = os.path.join(self.ctc_label_vocab_dir, 'ctc_detect_tags.txt')
        ctag_fp = os.path.join(self.ctc_label_vocab_dir,
                               'ctc_correct_tags.txt')

        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}

        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id

    def load_data(self) -> List[DataLoader]:

        # 加载train-dataset
        train_src_texts, train_trg_texts = self.load_texts_from_fp(
            self.train_fp)

        train_ds = DatasetCTC(
            in_model_dir=self.in_model_dir,
            src_texts=train_src_texts,
            trg_texts=train_trg_texts,
            max_seq_len=self.max_seq_len,
        )

        if self.dev_fp is not None:
            dev_src_texts, dev_trg_texts = self.load_texts_from_fp(
                self.dev_fp)
            dev_ds = DatasetCTC(
                in_model_dir=self.in_model_dir,
                src_texts=dev_src_texts,
                trg_texts=dev_trg_texts,
                max_seq_len=self.max_seq_len,
            )
        else:
            # 如果没有dev set,则从训练集切分
            _dev_size = max(int(len(train_ds) * self.dev_data_ratio), 1)
            _train_size = len(train_ds) - _dev_size
            train_ds, dev_ds = torch.utils.data.random_split(
                train_ds, [_train_size, _dev_size])

        if self.test_fp is not None:
            test_src_texts, test_trg_texts = self.load_texts_from_fp(
                self.test_fp)
            test_ds = DatasetCTC(
                in_model_dir=self.in_model_dir,
                src_texts=test_src_texts,
                trg_texts=test_trg_texts,
                max_seq_len=self.max_seq_len,
            )

        else:
            test_ds = None

        self._train_size = len(train_ds)
        self._dev_size = len(dev_ds)
        self._test_size = len(test_ds) if test_ds is not None else 0

        self._train_steps = ceil(
            self._train_size / self.batch_size)  # 训练总step num
        self._dev_steps = ceil(self._dev_size / self.batch_size)  # 训练总step num
        self._test_steps = ceil(
            self._test_size / self.batch_size)  # 训练总step num

        self.check_val_every_n_steps = ceil(
            self.check_val_every_n_epoch * self._train_steps)  # 每多少个step进行验证

        # 如果是分布式训练，则步数要除以总节点数
        self._train_steps = ceil(self._train_steps / self.ddp_nodes_num)
        self._dev_steps = ceil(self._dev_steps / self.ddp_nodes_num)
        self._test_steps = ceil(self._test_steps / self.ddp_nodes_num)

        if self.check_val_every_n_steps < 10:
            self.check_val_every_n_steps = 10

        logger.info('_train_size:{}'.format(self._train_size))
        logger.info('_dev_size:{}'.format(self._dev_size))
        logger.info('_test_size:{}'.format(self._test_size))

        logger.info('Total Steps of one epoch : {}'.format(self._train_steps))
        logger.info('Evaluation every {} steps'.format(
            self.check_val_every_n_steps))

        if self.ddp_local_rank != -1:
            # 如果使用分布式训练, 对train_ds进行DistributedSampler
            train_ds = torch.utils.data.dataloader.DataLoader(
                train_ds, sampler=DistributedSampler(train_ds), batch_size=self.batch_size, num_workers=4)

            dev_ds = torch.utils.data.dataloader.DataLoader(
                dev_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

            if test_ds is not None:
                test_ds = torch.utils.data.dataloader.DataLoader(
                    test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

        else:
            train_ds = torch.utils.data.dataloader.DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
            dev_ds = torch.utils.data.dataloader.DataLoader(
                dev_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
            if test_ds is not None:
                test_ds = torch.utils.data.dataloader.DataLoader(
                    test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return [train_ds, dev_ds, test_ds]

    def load_suite(self):
        "model"
        model = ModelingCtcBert.from_pretrained(
            self.in_model_dir)

        if self.freeze_embedding:
            embedding_name_list = ('embeddings.word_embeddings.weight',
                                   'embeddings.position_embeddings.weight',
                                   'embeddings.token_type_embeddings.weight')
            for named_para in model.named_parameters():
                named_para[1].requires_grad = False if named_para[
                    0] in embedding_name_list else True

        "optimizer"
        # bert常用权重衰减
        model_params = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in model_params
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.01
        }, {
            'params':
            [p for n, p in model_params if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        "scheduler"
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self._warmup_steps,
            num_training_steps=self._train_steps
        ) if self._warmup_steps != -1 else None
        return model, optimizer, scheduler

    def save_model(self, out_model_dir):
        "保存模型"
        model_to_save = self.model.module if hasattr(
            self.model, 'module') else self.model
        if self.ddp_local_rank in (-1, 0):
            if not os.path.exists(out_model_dir):

                os.mkdir(out_model_dir)
            model_to_save.save_pretrained(out_model_dir)
            self.tokenizer.save_pretrained(out_model_dir)
            logger.info('=========== New Model saved at {} ============='.format(
                out_model_dir))

    @staticmethod
    def fit_seed(random_seed_num):
        "固定随机种子 保证每次结果一样"
        np.random.seed(random_seed_num)
        torch.manual_seed(random_seed_num)
        torch.cuda.manual_seed_all(random_seed_num)
        torch.backends.cudnn.deterministic = True

    def train(self):
        """[summary]
        Args:
            wait_cuda_memory (bool, optional): []. Defaults to False.
        Returns:
            [type]: [description]
        """

        
        self.equip_cuda()
        best_eval_score = 0
        ith_early_stop_time = 0
        final_eval_scores_for_early_stop = []
        steps = ceil(self._train_size / self.batch_size)
        epoch_end_flag = False  # 每个epoch再验证
        for epoch in range(1, self.epochs + 1):
            if self.ddp_local_rank != -1:
                self.train_ds.sampler.set_epoch(epoch)
            if self.with_train_epoch_metric:
                epoch_preds, epoch_gold_labels = [], []
            else:
                epoch_preds, epoch_gold_labels = None, None
            epoch_c_loss = 0
            for step, batch_ds in track(enumerate(self.train_ds),
                                        description='Training',
                                        total=self._train_steps):
                step += 1

                # 训练过程可能有些许数据出错，跳过
                try:
                    batch_c_loss, batch_gold, batch_pred = self.train_step(
                        batch_ds, return_for_epoch_metric=self.with_train_epoch_metric)
                except RuntimeError as e:
                    logger.error('ignore training step error!!')
                    logger.exception(e)
                    continue

                if self.with_train_epoch_metric:
                    epoch_preds += batch_pred
                    epoch_gold_labels += batch_gold

                epoch_c_loss += batch_c_loss
                if (step % self.check_val_every_n_steps == 0 or epoch_end_flag) and self.ddp_local_rank in (-1, 0):
                    #  到达验证步数，则开始验证，保存模型，记录最大的dev指标
                    logger.info('[Start Evaluating]: local rank {}'.format(
                        self.ddp_local_rank))
                    epoch_end_flag = False
                    eval_epoch_c_loss, eval_epoch_precision, eval_epoch_recall, eval_epoch_f1 = self.evaluate(
                        dataset_type='dev')
                    log_text = '[Evaluating] Epoch {}/{}, Step {}/{}, ' \
                               'epoch_c_loss:{}, epoch_precision:{}, epoch_recall:{}, epoch_f1:{},  '
                    logger.info(
                        log_text.format(epoch, self.epochs, step, steps,
                                        eval_epoch_c_loss, eval_epoch_precision,
                                        eval_epoch_recall, eval_epoch_f1,
                                        ))
                    if self.test_ds is not None:

                        test_epoch_c_loss, test_epoch_precision, test_epoch_recall, test_epoch_f1 = self.evaluate(
                            dataset_type='test')

                        log_text = '[Testing] Epoch {}/{}, Step {}/{}, ' \
                            'epoch_c_loss:{}, epoch_precision:{}, epoch_recall:{}, epoch_f1:{}'
                        logger.info(
                            log_text.format(epoch, self.epochs, step, steps,
                                            test_epoch_c_loss, test_epoch_precision,
                                            test_epoch_recall, test_epoch_f1,
                                            ))

                    if eval_epoch_f1 >= 0:
                        #
                        if eval_epoch_f1 > best_eval_score:
                            best_eval_score = eval_epoch_f1
                            # 重置early stop次数
                            ith_early_stop_time = 0
                            final_eval_scores_for_early_stop = []
                        else:
                            # 验证集指标在下降，记录次数，为提前结束做准备。
                            ith_early_stop_time += 1
                            final_eval_scores_for_early_stop.append(
                                eval_epoch_f1)
                            if ith_early_stop_time >= self.early_stop_times:
                                logger.info(
                                    '[Early Stop], final eval_score:{}'.format(
                                        final_eval_scores_for_early_stop))
                                return
                        if self.test_ds is not None:
                            test_f1_str = str(round(test_epoch_f1 * 100,
                                                    2)).replace('.', '_') + '%'
                        else:
                            test_f1_str = 'None'
                        dev_f1_str = str(round(eval_epoch_f1 * 100,
                                               2)).replace('.', '_') + '%'
                        metric_str = 'epoch{},step{},testf1_{},devf1_{}'.format(epoch, step,
                                                                                test_f1_str, dev_f1_str)
                        saved_dir = os.path.join(
                            self.out_model_dir, metric_str)
                        if self.ddp_local_rank in (-1, 0):
                            self.save_model(saved_dir)

                        if eval_epoch_f1 >= 1:
                            # 验证集指标达到100%
                            logger.info(
                                'Devset f1-score has reached to 1.0, check testset f1')
                            if self.test_ds is not None and test_epoch_f1>=1:
                                logger.info(
                                'Testset f1-score has reached to 1.0, stop training')
                                return

            if self.with_train_epoch_metric:
                epoch_src = [self._keep_id_in_ctag]*len(epoch_src)
                (d_precision, d_recall, d_f1), (c_precision, c_recall, c_f1) = ctc_f1(
                    src_texts=[epoch_src], trg_texts=[epoch_gold_labels], pred_texts=[epoch_preds])

            else:
                epoch_precision, epoch_recall, epoch_f1 = None, None, None

            if self.ddp_local_rank in (-1, 0):
                logger.info('Epoch End..')
                epoch_end_flag = True
                log_text = '[Training epoch] Epoch {}/{},' \
                    'epoch_c_loss:{}, epoch_precision:{}, epoch_recall:{}, epoch_f1:{}'
                logger.info(
                    log_text.format(epoch, self.epochs, epoch_c_loss,
                                    epoch_precision, epoch_recall, epoch_f1))

        return 1

    def equip_cuda(self):

        if torch.cuda.is_available():
            self.model.cuda()
            # self.criterion.cuda()
            device_count = torch.cuda.device_count()
            devices_ids = list(range(device_count))
            if self.training_mode == 'dp' and device_count > 1:
                self.model = torch.nn.DataParallel(self.model,
                                                   device_ids=devices_ids)
                logger.info('DP training, use cuda list:{}'.format(
                    devices_ids))
            elif self.ddp_local_rank != -1:
                self.model = DDP(self.model, device_ids=[int(
                    self.ddp_local_rank)], output_device=int(self.ddp_local_rank), find_unused_parameters=True)
                logger.info('DDP training, use cuda list:{}'.format(
                    devices_ids))
            else:
                logger.info('Use single cuda to train')
        else:
            logger.info('Use cpu to train')

    def train_step(self, batch_ds, return_for_epoch_metric=True):

        self.model.train()

        if torch.cuda.is_available():
            for k, v in batch_ds.items():
                batch_ds[k] = v.cuda()

        self.optimizer.zero_grad()

        if self.amp and torch.cuda.is_available():
            # 混合精度模式
            with autocast():
                detect_outputs, correct_output, batch_loss = self.model(
                    input_ids=batch_ds['input_ids'],
                    attention_mask=batch_ds['attention_mask'],
                    token_type_ids=batch_ds['token_type_ids'],
                    detect_labels=batch_ds['d_tags'],
                    correct_labels=batch_ds['c_tags'],
                )
                batch_loss = batch_loss.mean()
            if self._max_grad_norm is None:
                self.scaler.scale(batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.scaler.scale(batch_loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._max_grad_norm)
                # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                self.scaler.step(self.optimizer)

                # Updates the scale for next iteration.
                self.scaler.update()
        else:
            # 常规模式
            detect_outputs, correct_output, batch_loss = self.model(
                input_ids=batch_ds['input_ids'],
                attention_mask=batch_ds['attention_mask'],
                token_type_ids=batch_ds['token_type_ids'],
                detect_labels=batch_ds['d_tags'],
                correct_labels=batch_ds['c_tags'],
            )
            batch_loss = batch_loss.mean()
            if self._max_grad_norm is None:
                batch_loss.backward()
                self.optimizer.step()
            else:
                batch_loss.backward()
                clip_grad_norm(self.model.parameters(), self._max_grad_norm)
                self.optimizer.step()

        # scheduler
        if self._warmup_steps != -1:
            self.scheduler.step()

        if return_for_epoch_metric:
            batch_gold = batch_ds['c_tags'].view(-1).tolist()
            batch_pred = torch.argmax(correct_output,
                                      dim=-1).view(-1).tolist()

            seq_true_idx = np.argwhere(batch_gold != self._loss_ignore_id)
            batch_gold = batch_gold[seq_true_idx].squeeze()
            batch_pred = batch_pred[seq_true_idx].squeeze()

            return batch_loss.item(), list(batch_gold), list(batch_pred)
        else:

            return batch_loss.item(),  None, None

    @torch.no_grad()
    def evaluate(self, dataset_type='dev'):
        # 分布式训练时, 外层调用前会确认节点为-1,0时, 才会做验证
        self.model.eval()
        epoch_loss = 0
        epoch_preds, epoch_gold_labels, epoch_src = [], [], []
        ds = self.test_ds if dataset_type == 'test' else self.dev_ds
        for batch_ds in ds:
            if torch.cuda.is_available():
                for k, v in batch_ds.items():
                    batch_ds[k] = v.cuda()
            if self.amp and torch.cuda.is_available():
                with autocast():
                    detect_outputs, correct_output, batch_loss = self.model(
                        input_ids=batch_ds['input_ids'],
                        attention_mask=batch_ds['attention_mask'],
                        token_type_ids=batch_ds['token_type_ids'],
                        detect_labels=batch_ds['d_tags'],
                        correct_labels=batch_ds['c_tags'],
                    )
            else:
                detect_outputs, correct_output, batch_loss = self.model(
                    input_ids=batch_ds['input_ids'],
                    attention_mask=batch_ds['attention_mask'],
                    token_type_ids=batch_ds['token_type_ids'],
                    detect_labels=batch_ds['d_tags'],
                    correct_labels=batch_ds['c_tags'],
                )
            batch_loss = batch_loss.mean()

            # correct

            batch_gold = batch_ds['c_tags'].view(-1).cpu().numpy()
            batch_pred = torch.argmax(correct_output,
                                      dim=-1).view(-1).cpu().numpy()
            batch_src = batch_ds['input_ids'].view(-1).cpu().numpy()

            seq_true_idx = np.argwhere(batch_gold != self._loss_ignore_id)  # 获取非pad部分的标签

            batch_gold = batch_gold[seq_true_idx].squeeze()
            batch_pred = batch_pred[seq_true_idx].squeeze()
            batch_src = batch_src[seq_true_idx].squeeze()

            epoch_src += list(batch_src)

            epoch_gold_labels += list(batch_gold)
            epoch_preds += list(batch_pred)

            epoch_loss += batch_loss.item()

            "因为输出和输入空间不一样，所以计算指标要对应输出空间，原字符对应输出空间的keep"
            epoch_src = [self._keep_id_in_ctag]*len(epoch_src)
        (d_precision, d_recall, d_f1), (c_precision, c_recall, c_f1) = ctc_f1(
            src_texts=[epoch_src], trg_texts=[epoch_gold_labels], pred_texts=[epoch_preds])

        return epoch_loss, c_precision, c_recall, c_f1
