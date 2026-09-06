from datasets import load_dataset

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0"
)

print(dataset)
print(dataset["train"].features)