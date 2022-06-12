import imp
import threading
import schedule
import time


class SimpleJob(threading.Thread):
    callback = None

    def __init__(self):
        threading.Thread.__init__(self)
        # self.start()

    def run(self):
        print("SimpleJob started")
        schedule.every(5).seconds.do(self.job)
        while True:
            schedule.run_pending()
            time.sleep(1)

    def job(self):
        print("SimpleJob job")
        if self.callback is not None:
            self.callback()

    def setCallback(self, callback):
        self.callback = callback
