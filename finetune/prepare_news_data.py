from datasets import load_dataset
import json
import random
import os

def main():
    print("Downloading ag_news dataset...")
    ds = load_dataset("fancyzhx/ag_news")
    
    classes = ["World", "Sports", "Business", "Sci/Tech"]
    options_str = ", ".join(classes)
    
    def make_example(example):
        text = example["text"].replace("\n", " ").strip()
        label = classes[example["label"]]
        
        prompt = (
            f"Classify the following news article into one of these categories: {options_str}.\n\n"
            f"Article: {text}\n\n"
            f"Options: {options_str}\n"
            f"Category:"
        )
        return {"prompt": prompt, "completion": f" {label}", "label": label}

    os.makedirs("data/news", exist_ok=True)
    
    print("Formatting and splitting data...")
    # use 10000 for train, 1000 for valid, 1000 for test to keep it small
    train_data = list(ds["train"])
    test_data = list(ds["test"])
    
    random.seed(42)
    random.shuffle(train_data)
    
    splits = {
        "train": [make_example(ex) for ex in train_data[:10000]],
        "valid": [make_example(ex) for ex in train_data[10000:11000]],
        "test": [make_example(ex) for ex in test_data[:1000]]
    }
    
    for split_name, examples in splits.items():
        with open(f"data/news/{split_name}.jsonl", "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(examples)} examples to data/news/{split_name}.jsonl")

if __name__ == "__main__":
    main()
