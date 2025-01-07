from train import train
from test import test

def main():
    results, model, tokenizer = train()
    print("Validation Accuracy:", results["eval_accuracy"])
    test(model, tokenizer)


if __name__ == "__main__":
    main()
