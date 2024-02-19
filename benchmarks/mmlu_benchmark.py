from datasets import load_dataset
from .base_benchmark import BaseBenchmark


class MMLUBenchmark(BaseBenchmark):
    def load_benchmark(self, language="it", **kwargs):
        self.benchmark = load_dataset('csv',
                         data_files={'test':'data/mmlu_ita/test.csv'},
                         split='test',
                        )