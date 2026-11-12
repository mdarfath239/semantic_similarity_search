from pathlib import Path
import joblib

folder = Path("models")

for file in folder.glob("*"):

    print(file.name, file.stat().st_size, "bytes")

for name in [
    "search_document_vectors.pkl",
    "search_document_metadata.pkl"
]:

    path = folder / name

    if path.exists():

        obj = joblib.load(path)

        print("\nFile:", name)
        print("Type:", type(obj))

        if hasattr(obj, "shape"):
            print("Shape:", obj.shape)

        if hasattr(obj, "columns"):
            print("Columns:", obj.columns.tolist())