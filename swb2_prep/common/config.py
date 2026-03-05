from pathlib import Path
import tomllib

def load_project_options(path: Path) -> dict:
    """
    Load a project_options.toml file and return its contents
    as a nested dictionary.

    Raises:
        FileNotFoundError
        tomllib.TOMLDecodeError
    """
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data