import json

from src.corrector import Corrector


def prepare_for_uploadfile(in_model_dir,
                           in_json_file, 
                           out_json_file='data/test_output.json'):
        
    json_data_list = json.load(open(in_json_file, 'r', encoding='utf-8'))
    src_texts = [ json_data['source'] for json_data in json_data_list]
    corrector = Corrector(in_model_dir=in_model_dir)
    pred_texts = corrector(texts=src_texts)
    output_json_data = [ {'id':json_data['id'], 'inference': pred_text} for json_data, pred_text in zip(json_data_list, pred_texts)]
    
    out_json_file = open(out_json_file, 'w', encoding='utf-8')
    json.dump(output_json_data, out_json_file, ensure_ascii=False, indent=4)
