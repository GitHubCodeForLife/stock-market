from algorithms.AlogorithmFactory import AlgorithmFactory
from helper.log.LogService import LogService

from jobs.ScheduleJob import ScheduleJob
from jobs.utils.FileWaiter import FileWaiter
import threading
time = 3


class TrainSchedule(ScheduleJob):
    criterias = None
    trainTimes = 0

    def __init__(self, criterias):
        super().__init__(time)
        self.criterias = criterias

    def doJob(self):
        # print(+ )        print(self.criterias['isTrain'])
        if self.criterias['isTrain'] == False:
            train_file = FileWaiter().getTrainFile(
                self.criterias['symbol'], self.criterias['algorithm'])
            model_file = FileWaiter().getModelFile(
                self.criterias['symbol'], self.criterias['algorithm'])

            algorithm = AlgorithmFactory().getAlgorithm(
                self.criterias['algorithm'])
            train, valid, dataset = algorithm.run_predict(
                train_file, model_file)
            return {'valid': valid, 'train': train, 'dataset': dataset}
        return None

    def setCriterias(self, criterias):
        self.criterias = criterias
