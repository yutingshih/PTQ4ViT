import os
import pandas as pd
import torch


def validate_filename(filename: str, existed: str = "append") -> str:
    if existed == "overwrite":
        pass
    elif existed == "append":
        pass
    elif existed == "keep_both":
        base, ext = os.path.splitext(filename)
        cnt = 1
        while os.path.exists(filename):
            filename = f"{base}-{cnt}{ext}"
            cnt += 1
    elif existed == "raise" and os.path.exists(filename):
        raise FileExistsError(f"{filename} already exists.")
    else:
        raise ValueError(f"Unknown value for 'existed': {existed}")
    return filename


def save_csv(
    df: pd.DataFrame, filename: str, verbose: bool = True, existed: str = "append"
) -> None:
    validate_filename(filename, existed)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if existed == "append":
        if os.path.exists(filename):
            df = pd.concat([pd.read_csv(filename, index_col=0), df])
    df.to_csv(filename)

    print(f"{filename} saved.")
    if verbose:
        print(df)
    return df


def save_model(
    model, filename: str, verbose: bool = True, existed: str = "keep_both"
) -> None:
    validate_filename(filename, existed)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    torch.save(model.state_dict(), filename)

    if verbose:
        print(f"Model saved at {filename}. ({os.path.getsize(filename) / 1e6} MB)")
    else:
        print(f"Model saved at {filename}.")
