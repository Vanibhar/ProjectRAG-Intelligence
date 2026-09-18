"""Build persisted legal retrieval artifacts. This command is intentionally offline."""

import argparse
from runtime.indexing import build_index
from runtime.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS and BM25 artifacts once.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    settings = Settings()
    build_index(settings, batch_size=args.batch_size)
    print(f"Saved production retrieval artifacts to {settings.artifacts_dir}")


if __name__ == "__main__":
    main()
