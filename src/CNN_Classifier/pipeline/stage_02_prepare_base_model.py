from CNN_Classifier.config import ConfigurationManager,ConfigurationManager2
from CNN_Classifier.components.prepare_base_model import PrepareBaseModel
from CNN_Classifier import logger

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config                    = ConfigurationManager2()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model        = PrepareBaseModel(config = prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()