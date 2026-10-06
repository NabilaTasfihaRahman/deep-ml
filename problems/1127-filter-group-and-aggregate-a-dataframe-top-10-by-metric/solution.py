import pandas as pd

def solution(df):
    df=df[df['status']=="completed"]
    df=df.groupby("region",as_index=False)['amount'].sum()
    return df.sort_values("amount",ascending=False).head(10)
