from algorithms.AlogorithmFactory import AlgorithmFactory
from helper.log.LogService import LogService

from jobs.ScheduleJob import ScheduleJob
from jobs.utils.FileWaiter import FileWaiter
import threading
time = 3


class PredictSchedule(ScheduleJob):
    criterias = None
    trainTimes = 0

    def __init__(self, criterias):
        super().__init__(time)
        self.criterias = criterias

    def doJob(self):
        if self.criterias['isTrain'] == False:
            criterias_copy = self.criterias.copy()
            train_file = FileWaiter().getTrainFile(
                self.criterias['symbol'], self.criterias['algorithm'], self.criterias['features'])
            model_file = FileWaiter().getModelFile(
                self.criterias['symbol'], self.criterias['algorithm'], self.criterias['features'])

            algorithm = AlgorithmFactory().getAlgorithm(
                self.criterias['algorithm'])
            prediction, dataset = algorithm.run_predict(
                train_file, model_file)
            return {'dataset': dataset,
                    'prediction': prediction,
                    'criterias': criterias_copy}
        return None

    def setCriterias(self, criterias):
        self.criterias = criterias
