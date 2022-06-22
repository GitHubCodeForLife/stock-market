from algorithms.LSTMAlgorithm import LSTMAlgorithm


class AlgorithmFactory:
    def __init__(self):
        pass

    def getAlgorithm(self, algorithm_type):
        # if algorithm_type == "LSTM":
        #     return LSTMAlgorithm()
        return LSTMAlgorithm()
