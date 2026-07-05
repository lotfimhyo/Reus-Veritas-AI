import asyncio

class Kernel:
    def __init__(self):
        self.loop = asyncio.get_event_loop()
        self.tasks = []

    def schedule(self, coro):
        t = self.loop.create_task(coro)
        self.tasks.append(t)
        return t

kernel = Kernel()
