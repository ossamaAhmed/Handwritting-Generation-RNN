
class TrainingConfig(object):
    def __init__(self):
        self.BATCH_SIZE = 30
        self.SEQUENCE_LENGTH = 250
        self.NUM_OF_LSTM_CELLS = 400
        self.NUM_OF_HIDDEN_LAYERS = 3
        self.NUM_OF_MIXTURES = 20
        self.NUM_OF_CHARS = 57
        self.TIME_STEP_INPUT_SIZE = 3
        self.TIME_STEP_TARGET_SIZE = 3  # prediction so MDN as unconditional
        self.output_keep_prob = 0.5  # not su
        self.U = 50 #? max num of chars in a seq
        self.LEN_OF_ALPHABET = 74 #double check
        self.W_NUM_OF_GAUSSIANS = 10
        self.inference = False
        self.epsilon_regularizer = 1e-8
        self.grad_clip = 3.
        self.learning_rate = 5e-4 #early stopping to be done
        self.EPOCHS = 50
        self.decay_rate = 0.95


class InferenceConfig(object):
    def __init__(self):
        self.BATCH_SIZE = 1
        self.SEQUENCE_LENGTH = 1
        self.NUM_OF_LSTM_CELLS = 400
        self.NUM_OF_HIDDEN_LAYERS = 3
        self.NUM_OF_MIXTURES = 20
        self.NUM_OF_CHARS = 57
        self.TIME_STEP_INPUT_SIZE = 3
        self.TIME_STEP_TARGET_SIZE = 3  # prediction so MDN as unconditional
        self.output_keep_prob = 0.5  # not su
        self.U = 50  # ? max num of chars in a seq
        self.LEN_OF_ALPHABET = 74  # double check
        self.W_NUM_OF_GAUSSIANS = 10
        self.inference = True
        self.epsilon_regularizer = 1e-8
        self.EPOCHS = 100
        self.decay_rate = 0.95
