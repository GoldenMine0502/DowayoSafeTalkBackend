
class DataConverter:
    def __init__(self, paths, result_path):
        if type(paths) == list:
            self.paths = paths
        else:
            self.paths = [paths]

        self.result_path = result_path
        self.data = []
        self.results = []

    def load_data(self):
        for path in self.paths:
            with open(path, "rt") as f:
                for line in f:
                    self.data.append(line.rstrip())

    def convert_all(self, convert_function):
        for d in self.data:
            result = convert_function(d)
            self.results.append(result)

    def save_all(self, save_function):
        with open(self.result_path, "wt") as f:
            for result in self.results:
                f.write(save_function(result) + "\n")

