from jobs.ScheduleJob import ScheduleJob
from jobs.utils.FileWaiter import FileWaiter
from modeltrainer.FactoryTrainer import FactoryTrainer
from predictor.FactoryPredictor import FactoryPredictor


time = 10


class TrainerSchedule(ScheduleJob):
    criterias = None

    def __init__(self, criterias):
        super().__init__(time)
        self.criterias = criterias

    def doJob(self):
        # symbol = self.criterias['symbol']
        print("TrainerSchedule doJob")
        train_file = FileWaiter().getTrainFile(
            self.criterias['symbol'], self.criterias['algorithm'])
        model_file = FileWaiter().getModelFile(
            self.criterias['symbol'], self.criterias['algorithm'])

        algorithm = self.criterias['algorithm']
        features = self.criterias['features']
        trainer = FactoryTrainer().getTrainer(algorithm, features)
        trainer.run(train_file, model_file)

        return None

    def setCriterias(self, criterias):
        self.criterias = criterias
