from algorithms.AlogorithmFactory import AlgorithmFactory
from helper.log.LogService import LogService
from jobs.BaseJob import BaseJob

from jobs.utils.FileWaiter import FileWaiter
import threading


class PredictSchedule(threading.Thread):
    _listeners = []
    criterias = None

    def __init__(self, criterias):
        threading.Thread.__init__(self)
        self.criterias = criterias

    def run(self):
        print("PredictSchedule doJob")
        criterias_copy = self.criterias.copy()
        train_file = FileWaiter().getTrainFile(
            self.criterias['symbol'], self.criterias['algorithm'], self.criterias['features'])
        model_file = FileWaiter().getModelFile(
            self.criterias['symbol'], self.criterias['algorithm'], self.criterias['features'])

        algorithm = AlgorithmFactory().getAlgorithm(
            self.criterias['algorithm'])
        prediction, dataset = algorithm.run_predict(
            train_file, model_file)
        product = {'dataset': dataset,
                   'prediction': prediction,
                   'criterias': criterias_copy}

        self.emit(product)

    def addListener(self, listener):
        self._listeners.append(listener)

    def emit(self, product):
        for callback in self._listeners:
            callback(product)
