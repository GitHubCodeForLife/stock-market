import threading
import schedule
import time

from helper.log.LogService import LogService


class ScheduleJob(threading.Thread):
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
        print("Schedule job")
        product = self.doJob()
        self.emit(product)

    def doJob(self):
        print("Schedule doJob")
        return None

    def setTime(self, time):
        self._time = time

    def addListener(self, listener):
        self._listeners.append(listener)

    def emit(self, product):
        for callback in self._listeners:
            callback(product)
