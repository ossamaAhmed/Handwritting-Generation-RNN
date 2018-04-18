
class TrainingConfig(object):
    def __init__(self):
        self.BATCH_SIZE = 25
        self.SEQUENCE_LENGTH = 300
        self.NUM_OF_LSTM_CELLS = 400
        self.NUM_OF_HIDDEN_LAYERS = 3
        self.NUM_OF_MIXTURES = 20
        self.TIME_STEP_INPUT_SIZE = 3  # first being the cut boolean
        self.TIME_STEP_TARGET_SIZE = 3  # same as input
        self.momentum = 0.9
        self.decay = 0.95
        self.output_keep_prob = 0.5
        self.eps_lower = 1e-20
        self.eps_upper = 1e+20
        self.inference = False
        self.grad_clip = 3.
        self.learning_rate = 5e-4 #early stopping to be done
        self.EPOCHS = 100
        self.decay_rate = 0.95



class InferenceConfig(object):
    def __init__(self):
        #remove training parameters later on from the model as well
        self.BATCH_SIZE = 1
        self.SEQUENCE_LENGTH = 1
        self.NUM_OF_LSTM_CELLS = 400
        self.NUM_OF_HIDDEN_LAYERS = 3
        self.NUM_OF_MIXTURES = 20
        self.TIME_STEP_INPUT_SIZE = 3  # first being the cut boolean
        self.TIME_STEP_TARGET_SIZE = 3  # same as input
        self.momentum = 0.9
        self.decay = 0.95
        self.output_keep_prob = 0.5
        self.eps_lower = 1e-20
        self.eps_upper = 1e+20
        self.inference = True
        self.grad_clip = 3.
        self.learning_rate = 5e-4 #early stopping to be done
        self.EPOCHS = 100
        self.decay_rate = 0.95
