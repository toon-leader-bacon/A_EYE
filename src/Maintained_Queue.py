import torch


import queue


class Maintained_Queue():
    def __init__(self, maxSize: int):
        self.maxSize = maxSize
        self.queue = queue.Queue(maxsize=maxSize)

    def push(self, item):
        if self.queue.full():
            self.queue.get()
        self.queue.put(item)

    def len(self):
        return self.queue.qsize()

    def average_as_tensor(self):
        concat = torch.stack(list(self.queue.queue))
        return torch.mean(concat, dim=0)