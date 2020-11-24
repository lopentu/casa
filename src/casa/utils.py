from pathlib import Path

def get_data_path():
    base_path = Path(__file__).parent
    return (base_path / "../../data").resolve().absolute()