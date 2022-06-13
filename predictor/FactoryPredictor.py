from predictor.NSEpredictor import NSEpredictor


class FactoryPredictor:
    predictors = [NSEpredictor(), NSEpredictor()]

    @staticmethod
    def getPredictor(algorithm, feature):
        # if algorithm == "LSTM":
        #     return NSEpredictor()
        # elif algorithm == "RNN":
        #     return NSEpredictor()
        # elif algorithm == "CNN":
        #     return NSEpredictor()
        # elif algorithm == "MLP":
        #     return NSEpredictor()
        # else:
        #     return None
        return FactoryPredictor.predictors[0]
