from datasets import Dataset

# ds = Dataset.from_file("data/gsm8k-genesys/train/cache-c46a9c299e4785d1.arrow")
ds = Dataset.from_file("data/gsm8k-genesys/train/data-00000-of-00001.arrow")
print(ds[0])