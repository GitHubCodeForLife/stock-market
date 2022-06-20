from jobs.ScheduleJob import ScheduleJob
from jobs.utils.FileWaiter import FileWaiter
from modeltrainer.FactoryTrainer import FactoryTrainer
from predictor.FactoryPredictor import FactoryPredictor


time = 3


class PredictorSchedule(ScheduleJob):
    criterias = None

    def __init__(self, criterias):
        super().__init__(time)
        self.criterias = criterias

    def doJob(self):
        # symbol = self.criterias['symbol']
        print("PredictorSchedule doJob")
        train_file = FileWaiter().getTrainFile(
            self.criterias['symbol'], self.criterias['algorithm'])
        model_file = FileWaiter().getModelFile(
            self.criterias['symbol'], self.criterias['algorithm'])

        algorithm = self.criterias['algorithm']
        features = self.criterias['features']

        predictor = FactoryPredictor().getPredictor(algorithm, features)
        train, valid, dataset = predictor.run(train_file, model_file)
        return {'valid': valid, 'train': train, 'dataset': dataset}

    def setCriterias(self, criterias):
        self.criterias = criterias
