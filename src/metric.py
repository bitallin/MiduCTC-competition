import json
from difflib import SequenceMatcher

from src import logger


def f1(precision, recall):
    if precision + recall == 0:
        return 0
    return round(2 * precision * recall / (precision + recall), 4)


def compute_label_nums(src_text, trg_text, pred_text, log_error_to_fp=None):
    assert len(src_text) == len(trg_text) == len(
        pred_text), 'src_text:{}, trg_text:{}, pred_text:{}'.format(src_text, trg_text, pred_text)
    pred_num, detect_num, correct_num, ref_num = 0, 0, 0, 0

    for j in range(len(trg_text)):
        src_char, trg_char, pred_char = src_text[j], trg_text[j], pred_text[j]
        if src_char != trg_char:
            ref_num += 1
            if src_char != pred_char:
                detect_num += 1
            elif log_error_to_fp is not None and pred_char != trg_char and pred_char == src_char:
                log_text = '漏报\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    src_text, trg_text, src_char, trg_char, pred_char, j)
                log_error_to_fp.write(log_text)

        if src_char != pred_char:
            pred_num += 1
            if pred_char == trg_char:
                correct_num += 1
            elif log_error_to_fp is not None and pred_char != trg_char and src_char == trg_char:
                log_text = '误报\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    src_text, trg_text, src_char, trg_char, pred_char, j)
                log_error_to_fp.write(log_text)
            elif log_error_to_fp is not None and pred_char != trg_char and src_char != trg_char:
                log_text = '错报(检对报错)\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    src_text, trg_text, src_char, trg_char, pred_char, j)
                log_error_to_fp.write(log_text)

    return (pred_num, detect_num, correct_num, ref_num)


def ctc_f1(src_texts, trg_texts, pred_texts, log_error_to_fp=None):
    """训练过程中字级别序列标注任务的F1计算

    Args:
        src_texts ([type]): [源文本]
        trg_texts ([type]): [目标文本]
        pred_texts ([type]): [预测文本]
        log_error_to_fp : 文本路径

    Returns:
        [type]: [description]
    """
    if isinstance(src_texts, str):
        src_texts = [src_texts]
    if isinstance(trg_texts, str):
        trg_texts = [trg_texts]
    if isinstance(pred_texts, str):
        pred_texts = [pred_texts]
    lines_length = len(trg_texts)
    assert len(src_texts) == lines_length == len(
        pred_texts), 'keep equal length'
    all_pred_num, all_detect_num, all_correct_num, all_ref_num = 0, 0, 0, 0
    if log_error_to_fp is not None:
        f = open(log_error_to_fp, 'w', encoding='utf-8')
        f.write('type\tsrc_text\ttrg_text\tsrc_char\ttrg_char\tpred_char\tchar_index\n')
    else:
        f = None

    all_nums = [compute_label_nums(src_texts[i], trg_texts[i], pred_texts[i], f)
                for i in range(lines_length)]
    if log_error_to_fp is not None:
        f.close()
    for i in all_nums:
        all_pred_num += i[0]
        all_detect_num += i[1]
        all_correct_num += i[2]
        all_ref_num += i[3]

    d_precision = round(all_detect_num/all_pred_num,
                        4) if all_pred_num != 0 else 0
    d_recall = round(all_detect_num/all_ref_num, 4) if all_ref_num != 0 else 0
    c_precision = round(all_correct_num/all_pred_num,
                        4) if all_pred_num != 0 else 0
    c_recall = round(all_correct_num/all_ref_num, 4) if all_ref_num != 0 else 0

    d_f1, c_f1 = f1(d_precision, d_recall), f1(c_precision, c_recall)

    logger.info('====== [Char Level] ======')
    logger.info('d_precsion:{}%, d_recall:{}%, d_f1:{}%'.format(
        d_precision*100, d_recall*100, d_f1*100))
    logger.info('c_precsion:{}%, c_recall:{}%, c_f1:{}%'.format(
        c_precision*100, c_recall*100, c_f1*100))
    logger.info('error_char_num: {}'.format(all_ref_num))
    return (d_precision, d_recall, d_f1), (c_precision, c_recall, c_f1)


def ctc_comp_f1_sentence_level(src_texts, pred_texts, trg_texts):
    "计算负样本的 句子级 纠正级别 F1"
    correct_ref_num, correct_pred_num, correct_recall_num, correct_f1 = 0, 0, 0, 0
    for src_text, pred_text, trg_text in zip(src_texts, pred_texts, trg_texts):
        if src_text != pred_text:
            correct_pred_num += 1
        if src_text != trg_text:
            correct_ref_num += 1
        if src_text != trg_text and pred_text == trg_text:
            correct_recall_num += 1

    assert correct_ref_num > 0, '文本中未发现错误，无法计算指标，该指标只计算含有错误的样本。'

    correct_precision = 0 if correct_recall_num == 0 else correct_recall_num / correct_pred_num
    correct_recall = 0 if correct_recall_num == 0 else correct_recall_num / correct_ref_num
    correct_f1 = f1(correct_precision, correct_recall)

    return correct_precision, correct_recall, correct_f1


def ctc_comp_f1_token_level(src_texts, pred_texts, trg_texts):
    "字级别，负样本 检测级别*0.8+纠正级别*0.2 f1"
    def compute_detect_correct_label_list(src_text, trg_text):
        detect_ref_list, correct_ref_list = [], []
        diffs = SequenceMatcher(None, src_text, trg_text).get_opcodes()
        for (tag, src_i1, src_i2, trg_i1, trg_i2) in diffs:

            if tag == 'replace':
                for count, src_i in enumerate(range(src_i1, src_i2)):
                    trg_token = trg_text[trg_i1+count]
                    detect_ref_list.append(src_i)
                    correct_ref_list.append((src_i, trg_token))

            elif tag == 'delete':
                trg_token = 'D'*(src_i2-src_i1)
                detect_ref_list.append(src_i1)
                correct_ref_list.append((src_i1, trg_token))

            elif tag == 'insert':
                trg_token = trg_text[trg_i1:trg_i2]
                detect_ref_list.append(src_i1)
                correct_ref_list.append((src_i1, trg_token))

        return detect_ref_list, correct_ref_list

    # 字级别
    detect_ref_num, detect_pred_num, detect_recall_num, detect_f1 = 0, 0, 0, 0
    correct_ref_num, correct_pred_num, correct_recall_num, correct_f1 = 0, 0, 0, 0

    for src_text, pred_text, trg_text in zip(src_texts, pred_texts, trg_texts):
        # 先统计检测和纠正标签
        detect_pred_list, correct_pred_list = compute_detect_correct_label_list(
            src_text, pred_text)
        detect_ref_list, correct_ref_list = compute_detect_correct_label_list(
            src_text, trg_text)

        detect_ref_num += len(detect_ref_list)
        detect_pred_num += len(detect_pred_list)
        detect_recall_num += len(set(detect_ref_list)
                                 & set(detect_pred_list))

        correct_ref_num += len(correct_ref_list)
        correct_pred_num += len(correct_pred_list)
        correct_recall_num += len(set(correct_ref_list)
                                  & set(correct_pred_list))

    assert correct_ref_num > 0, '文本中未发现错误，无法计算指标，该指标只计算含有错误的样本。'

    detect_precision = 0 if detect_pred_num == 0 else detect_recall_num / detect_pred_num
    detect_recall = 0 if detect_ref_num == 0 else detect_recall_num / detect_ref_num

    correct_precision = 0 if detect_pred_num == 0 else correct_recall_num / correct_pred_num
    correct_recall = 0 if detect_ref_num == 0 else correct_recall_num / correct_ref_num

    detect_f1 = f1(detect_precision, detect_recall)
    correct_f1 = f1(correct_precision, correct_recall)

    final_f1 = detect_f1*0.8+correct_f1*0.2

    return final_f1, [detect_precision, detect_recall, detect_f1], [correct_precision, correct_recall, correct_f1]


def final_f1_score(src_texts,
                   pred_texts,
                   trg_texts,
                   log_fp='logs/f1_score.log'):
    """"最终输出结果F1计算，综合了句级F1和字级F1"

    Args:
        src_texts (_type_): 源文本
        pred_texts (_type_): 预测文本
        trg_texts (_type_): 目标文本
        log_fp (str, optional): _description_. Defaults to 'logs/f1_score.log'.

    Returns:
        _type_: _description_
    """
    
    
    
    token_level_f1, detect_metrics, correct_metrcis = ctc_comp_f1_token_level(
        src_texts, pred_texts, trg_texts)
    sent_level_f1, sent_level_p, sent_level_r = ctc_comp_f1_sentence_level(
        src_texts, pred_texts, trg_texts)
    final_f1 = round(0.8*token_level_f1 + sent_level_f1*0.2, 4)

    json_data = {

        'token_level:[detect_precision, detect_recall, detect_f1]': detect_metrics,
        'token_level:[correct_precision, correct_recall, correct_f1] ': correct_metrcis,
        'token_level:f1': token_level_f1,

        'sentence_level:[correct_precision, correct_recall]': [sent_level_p, sent_level_r],
        'sentence_level:f1': sent_level_f1,

        'final_f1': final_f1
    }
    _log_fp = open(log_fp, 'w', encoding='utf-8')
    json.dump(json_data, _log_fp, indent=4)
    logger.info('final f1:{}'.format(final_f1))
    logger.info('f1 logfile saved at:{}'.format(log_fp))
    return final_f1