import json
from src.corrector import Corrector
from src.metric import final_f1_score


def evaluate(in_model_dir,json_data_file, log_fp='logs/f1_score.log'):
    """输入模型目录，数据， 计算模型在该数据下的指标

    """
    
    json_data = json.load(open(json_data_file, 'r', encoding='utf-8'))
    src_texts, trg_texts = [], []
    for line in json_data:
        src_texts.append(line['source'])
        trg_texts.append(line['target'])
    
    corrector = Corrector(in_model_dir=in_model_dir)
    pred_texts = corrector(texts=src_texts)
    f1_score = final_f1_score(src_texts=src_texts, 
                              pred_texts=pred_texts,
                              trg_texts=trg_texts,
                              log_fp=log_fp)
    
    return f1_score