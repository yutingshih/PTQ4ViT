import sys

import pandas as pd


def transform(df, verbose=False):
    df = df.pivot(index='model', columns=['config', 'bit'], values='accuracy')
    df.columns = ['PTQ4ViT W8A8', 'BasePTQ W8A8', 'PTQ4ViT W6A6', 'BasePTQ W6A6']
    df = df[['BasePTQ W8A8', 'PTQ4ViT W8A8', 'BasePTQ W6A6', 'PTQ4ViT W6A6']]
    df.index.name = 'Model'
    df = df.iloc[::-1]
    if verbose:
        print(df)
    return df.to_markdown()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("Please provide a csv file to convert to markdown")

    df = pd.read_csv(sys.argv[1])
    md = transform(df)
    print(md)
