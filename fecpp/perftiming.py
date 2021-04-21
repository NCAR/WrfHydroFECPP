import time
from contextlib import ContextDecorator


class performance_timing(ContextDecorator):
    def __init__(self, label=None):
        self.label = label

    def __enter__(self):
        self.start_time = time.monotonic()
        return self

    def __exit__(self, *exc):
        stop_time = time.monotonic() - self.start_time
        print(f'Elapsed time {f"for {self.label}" if self.label is not None else ""}= {stop_time:.5} seconds')
        return False

