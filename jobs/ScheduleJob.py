import threading
import schedule
import time


class BaseJob(threading.Thread):
    _listeners = []
    _time = 10

    def __init__(self):
        threading.Thread.__init__(self)

    def __init__(self, time):
        self._time = time
        threading.Thread.__init__(self)

    def run(self):
        schedule.every(self._time).seconds.do(self.job)
        while True:
            schedule.run_pending()
            time.sleep(1)

    def job(self):
        product = self.doJob()
        for callback in self._listeners:
            callback(product)

    def doJob(self):
        print("BaseJob doJob")
        return None

    def addListener(self, listener):
        self._listeners.append(listener)

    def setTime(self, time):
        self._time = time
