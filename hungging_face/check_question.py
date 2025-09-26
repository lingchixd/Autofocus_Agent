from datasets import load_dataset
from pprint import pprint

ds = load_dataset("jnirschl/uBench", split="test")
pprint(ds[0]["questions"])
