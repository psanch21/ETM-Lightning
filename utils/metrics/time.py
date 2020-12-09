import time
class Timer:

    def __init__(self):
        self.timer_dict = {}



    def tic(self, name):
        self.timer_dict[name] = time.time()

    def toc(self, name):
        assert  name  in self.timer_dict
        elapsed = time.time() - self.timer_dict[name]
        del self.timer_dict[name]
        return elapsed
