import pandas as pd

path = '/Users/philadlamini/Downloads/processed.csv'
df = pd.read_csv(path)

# Validation checks
assert df.shape[1] == 2, "CSV does not have exactly 2 columns"

for i, (seq, target) in enumerate(zip(df.iloc[:, 0], df.iloc[:, 1]), start=2):  # start=2 accounts for header row
    assert len(seq) == 1000, f"Sequence length is not 1000 at line {i}"
    assert len(target) == 39, f"Target length is not 39 at line {i}"
    assert all(c in '01' for c in target), f"Target contains non-binary characters at line {i}"

print("All sequences and targets are valid.")
