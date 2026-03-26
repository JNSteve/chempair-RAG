import os
import subprocess
import tarfile
import uvicorn

RAG_STORAGE = os.environ.get("RAG_STORAGE", "/data/rag_storage")
RELEASE_URL = "https://github.com/JNSteve/chempair-RAG/releases/download/v1.0/rag_storage.tar.gz"

def ensure_data():
    """Download and extract rag_storage if the volume is empty."""
    marker = os.path.join(RAG_STORAGE, "vdb_entities.json")
    if os.path.exists(marker):
        print(f"Data found at {RAG_STORAGE}, skipping download.")
        return

    os.makedirs(RAG_STORAGE, exist_ok=True)
    tarball = "/tmp/rag_storage.tar.gz"

    print(f"Downloading knowledge graph data from GitHub release...")
    subprocess.run(["curl", "-L", "-o", tarball, RELEASE_URL], check=True)

    print(f"Extracting to {RAG_STORAGE}...")
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(path=RAG_STORAGE)

    os.remove(tarball)
    print("Data ready.")


if __name__ == "__main__":
    ensure_data()
    # Set env var so server.py can find it
    os.environ["RAG_STORAGE"] = RAG_STORAGE
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
