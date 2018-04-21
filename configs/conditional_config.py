
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
        self.LEN_OF_ALPHABET = 65 #double check
        self.W_NUM_OF_GAUSSIANS = 10
        self.inference = False
        # self.TIME_STEP_TARGET_SIZE = 3  # same as input
        # self.momentum = 0.9
        # self.decay = 0.95
        # self.output_keep_prob = 0.5
        # self.eps_lower = 1e-20
        # self.eps_upper = 1e+20
        # self.inference = False
        self.grad_clip = 3.
        self.learning_rate = 1e-3 #early stopping to be done
        self.EPOCHS = 100
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
        self.LEN_OF_ALPHABET = 65  # double check
        self.W_NUM_OF_GAUSSIANS = 10
        self.inference = True
        # self.TIME_STEP_TARGET_SIZE = 3  # same as input
        # self.momentum = 0.9
        # self.decay = 0.95
        # self.output_keep_prob = 0.5
        # self.eps_lower = 1e-20
        # self.eps_upper = 1e+20
        # self.inference = False
        self.grad_clip = 3.
        self.learning_rate = 1e-3  # early stopping to be done
        self.EPOCHS = 100
        self.decay_rate = 0.95