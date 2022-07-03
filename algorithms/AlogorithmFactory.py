from algorithms.LSTMAlgorithm import LSTMAlgorithm
from algorithms.XGBoostAlgorithm import XGBoostAlgorithm


class AlgorithmFactory:
    def __init__(self):
        pass

    def getAlgorithm(self, algorithm_type):
        if algorithm_type == "LSTM":
            return LSTMAlgorithm()
        elif algorithm_type == "XGboost":
            return XGBoostAlgorithm()

        return LSTMAlgorithm()
