import pandas as pd
import sys
import os

dataname = "quora"
datasets = ["train", "dev", "test"]

for ds in datasets:
    inpath = os.path.join("data","quora",ds + ".tsv")
    df = pd.read_csv(inpath,
                     sep=",")
    df.to_csv(inpath, columns=["id", "text1", "text2", "label"], index=False, sep="\t")