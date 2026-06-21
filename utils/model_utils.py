from pathlib import Path

def extract_embedding_name(path: str) -> str:
    filename = Path(path).stem
    if 'embeddings_' in filename:
        name = filename.replace('embeddings_', '').replace('_api', '')
        name = name.replace('@', '_')
        return name
    else:
        raise ValueError("Unknown unknown_embedding:", path)


def extract_split_method(path: str) -> str:
    filename = Path(path).stem
    if 'stratified' in filename:
        return 'stratified'
    elif 'morphological' in filename:
        return 'morphological'
    elif 'community' in filename:
        return 'community'
    elif 'cluster' in filename:
        return 'cluster'
    raise ValueError("Split method not detected in", path)


def extract_dataset(path: str) -> str:
    filename = Path(path).stem.lower()
    if 'plutchik' in filename:
        return 'plutchik'
    elif 'vad' in filename:
        return 'vad'
    raise ValueError(f"Could not extract dataset from path: {path}. Expected 'vad' or 'plutchik' in filename.")
