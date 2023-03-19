from pathlib import Path


def get_cache_file(filename: str, remove: bool = False):
    """
    Returns a name in the cache folder `~/.onnx-array-api`.

    :param filename: filename
    :param remove: remove if exists
    :return: full filename
    """
    home = Path.home()
    folder = home / ".onnx-array-api"
    if not folder.exists():
        folder.mkdir()
    name = folder / filename
    if name.exists():
        name.unlink()
    return name
