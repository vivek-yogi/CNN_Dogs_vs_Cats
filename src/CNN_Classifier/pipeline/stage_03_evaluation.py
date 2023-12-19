from CNN_Classifier.config import ConfigurationManager
from CNN_Classifier.components.evaluation  import Evaluation
from CNN_Classifier import logger

class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config     = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()
