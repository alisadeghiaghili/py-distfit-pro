"""Canonical executable documentation example with no unimplemented API claim."""

DATA = (1.0, 2.0, 3.0)
EXAMPLE_RESULT = {"count": len(DATA), "mean": sum(DATA) / len(DATA)}

if __name__ == "__main__":
    print(EXAMPLE_RESULT)
