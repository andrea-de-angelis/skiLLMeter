# skiLLMeter

skiLLMeter is a tool to test the skills of Language Models (no-chat version). It works forcing the model to generate a given string and computing the loss associated to the generation.

It supports the following benchmarks:
- [X] xcopa
- [ ] truthfulQA
- [ ] MMLU

The metric computed is the accuracy.

# Install

You can create a virtual environment using `conda` or `virtualenv`.

With `conda`:

`conda create --name skillmeter --python=3.8.16 --file requirements.txt`

With `virtualenv`:

`pip install -r requirements.txt`

# Run

**Pre-requisite:** add your HuggingFace token into the `conf.env` file.

To run the script you can do as follows:

`python main.py --model-id meta-llama/Llama-2-7b-hf --labels "<toxic-no> <linguistic_quality-high>"`

Please note that the `labels` parameter is optional: if you specify it, the string will be added as a prefix to text, and separeted from the text with a pipe (|); if you don't specify it, only the text will be used for the prompt.

# Output

The script will generate two files in the `outputs` folder:
- a log file, containing information about the experiment and the final metric
- a csv file, containing the prediction for the specified benchmark