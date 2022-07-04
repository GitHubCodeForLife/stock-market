from algorithms.LSTMAlgorithm import LSTMAlgorithm
from algorithms.XGBoostAlgorithm import XGBoostAlgorithm
from algorithms.RNNAlgorithm import RNNAlgorithm


class AlgorithmFactory:
    def __init__(self):
        pass

    def getAlgorithm(self, algorithm_type):
        if algorithm_type == "LSTM":
            return LSTMAlgorithm()
        elif algorithm_type == "XGboost":
            return XGBoostAlgorithm()
        elif algorithm_type == "RNN":
            return RNNAlgorithm()

        return LSTMAlgorithm()
