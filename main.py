import torch
import logging
import argparse
import configparser
import huggingface_hub
from evaluation.evaluation import compute_accuracy
from benchmarks.xcopa_benchmark import XCOPABenchmark
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformations.xcopa_transformation import XCOPATransformation


def load_model_from_hf(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to('cuda')
    
    return tokenizer, model


def setup_logger(model_id, benchmark, labels):
    model_id = model_id.replace("/", "__")
    labels = labels.replace(" ", "__")
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(message)s",
        handlers = [
            logging.FileHandler(f"outputs/{model_id}_xcopa_{labels}.log"),
            logging.StreamHandler()
        ]
    )
    

def setup_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-id', required=True)
    parser.add_argument('--labels')

    return parser


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not args.labels:
        labels = 'no_labels'
    else:
        labels = args.labels.replace(" ", "__")
        
    setup_logger(args.model_id, "xcopa", labels)
    
    config = configparser.ConfigParser()
    config.read("conf.env")
    hf_loging_token = config["HF"]["HUGGINGFACE_LOGIN_TOKEN"]
    huggingface_hub.login(hf_loging_token)
    
    logging.info(f"Model ID: {args.model_id}")
    logging.info("Benchmark: xcopa")
    logging.info(f"Labels: {args.labels}")

    tokenizer, model = load_model_from_hf(args.model_id)
    
    xcopa_dataset = XCOPABenchmark("xcopa")
    xcopa_dataset.load_benchmark(language="it")
    
    xcopa_transformation = XCOPATransformation(xcopa_dataset.get_data())
    xcopa_transformation.transform()
    
    transformed_xcopa_data = xcopa_transformation.get_transformed_data()

    logging.info(f"Number of samples: {len(transformed_xcopa_data)}")
    logging.info(transformed_xcopa_data.head())
    
    xcopa_transformation.predict(model, tokenizer, labels=args.labels)
    
    model_id = args.model_id.replace("/", "__")
    xcopa_transformation.get_transformed_data().to_csv(f"outputs/{model_id}_xcopa_{labels}.csv", index=False)
    accuracy = compute_accuracy(xcopa_transformation.get_transformed_data())
    logging.info(f"Accuracy: {accuracy}")