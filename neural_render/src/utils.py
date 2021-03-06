from pathlib import Path
import sys


def mkdirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class StderrHolder():
    def __init__(self, file):
        self.file = file
        self.stderr = sys.stderr
        sys.stderr = self

    def write(self, data):
        self.file.write(data)
        self.stderr.write(data)

    def flush(self):
        self.file.flush()


class PrintAndLog():
    def __init__(self, file_name):
        super().__init__()

        self.file = open(file_name, 'w')
        self.stdout = sys.stdout
        sys.stdout = self

        self.stderr_holder = StderrHolder(self.file)

    def __del__(self):
        sys.stderr = self.stderr_holder.stderr
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
