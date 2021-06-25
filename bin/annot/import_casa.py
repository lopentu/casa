from pathlib import Path
base_path = Path(__file__).parent
src_path = base_path / "../../src"
import sys
sys.path.append(str(src_path))

import casa
import casa.annot as cano
import logging
logging.basicConfig(level="INFO", format="[%(levelname)s] %(asctime)s %(name)s: %(message)s")