import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_publication_consistency.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_publication_consistency", CHECKER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_result_artifacts_match_canonical_publication_values():
    checker = load_checker()
    checker.check_result_files()


def test_publication_documents_use_conservative_consistent_framing():
    checker = load_checker()
    checker.check_docs()
