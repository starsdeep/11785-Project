import pandas as pd

def train_test_split(df, test_size=200):
    if isinstance(test_size, float):
        test_size = int(len(df) * test_size)

    df = df.sample(frac=1)
    df_test = df.iloc[:test_size]
    df_train = df.iloc[test_size:]
    return df_train, df_test


if __name__ == '__main__':
    df = pd.read_pickle('/mnt/truenas/scratch/yikangliao/dataset/LIRIS_D2/training-validation-df.pickle')
    df_train, df_valid = train_test_split(df, 0.2)
    df_train.to_pickle('/mnt/truenas/scratch/yikangliao/dataset/LIRIS_D2/train-df.pickle')
    df_valid.to_pickle('/mnt/truenas/scratch/yikangliao/dataset/LIRIS_D2/valid-df.pickle')
