from datasets import load_dataset
from .base_benchmark import BaseBenchmark


class XCOPABenchmark(BaseBenchmark):
    def load_benchmark(self, language="it", **kwargs):
        self.benchmark = load_dataset("xcopa", language)