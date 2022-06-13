from modeltrainer.LSTMTrainer import LSTMTrainer


class FactoryTrainer:
    def __init__(self):
        pass

    @staticmethod
    def getTrainer(algorithm, feature):
        # if algorithm == "LSTM":
        #     return LSTMTrainer()
        # elif algorithm == "RNN":
        #     return RNNTrainer()
        # elif algorithm == "CNN":
        #     return CNNTrainer()
        # elif algorithm == "MLP":
        #     return MLPTrainer()
        # else:
        #     return None
        return LSTMTrainer()
