from configs.conditional_config import TrainingConfig
from models.conditional_model import Model

config = TrainingConfig()
my_model = Model(config)
my_model.build_model()
