
from datetime import datetime


file = "./static/data/log.txt"


class LogService:
    def __init__(self):
        pass

    @staticmethod
    def logToFile(message):
        with open(file, "w") as f:
            f.write(str(datetime.now()) + ": ")
            f.write(str(message) + "\n")

    @staticmethod
    def logAppendToFile(message):
        with open(file, "a") as f:
            # write current time
            f.write(str(datetime.now()) + ": ")
            f.write(str(message) + "\n")
