
from typing import List

from src.baseline.predictor import PredictorCtc


class Corrector:
    def __init__(self, in_model_dir:str):
        """_summary_

        Args:
            in_model_dir (str): 训练好的模型目录
        """
        self._predictor = PredictorCtc(
        in_model_dir=in_model_dir,
        ctc_label_vocab_dir='src/baseline/ctc_vocab',
        use_cuda=True,
        cuda_id=None,
    )
        
    
    def __call__(self, texts:List[str]) -> List[str]:
        pred_outputs = self._predictor.predict(texts)
        pred_texts = [PredictorCtc.output2text(output) for output in pred_outputs]
        return pred_texts

    