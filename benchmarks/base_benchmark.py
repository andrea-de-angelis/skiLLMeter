class BaseBenchmark:
    def __init__(self, benchmark_name):
        self.benchmark_name = benchmark_name
        self.benchmark = None

    def load_benchmark(self, **kwargs):
        raise NotImplementedError("load_benchmark method must be implemented in the derived class.")
    
    
    def get_data(self):
        if self.benchmark is None:
            raise ValueError("Benchmark has not been loaded. Call load_benchmark first.")
        return self.benchmark