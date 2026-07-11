import glob


def resolve_paths(patterns):
    paths = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        paths.extend(matches if matches else [p])
    if not paths:
        raise SystemExit(f"No files matched: {patterns}")
    return paths
