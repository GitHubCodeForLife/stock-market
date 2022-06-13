import threading


class BaseJob(threading.Thread):
    _listeners = []

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        return None

    def emit(self, product):
        for callback in self._listeners:
            callback(product)

    def addListener(self, listener):
        self._listeners.append(listener)
