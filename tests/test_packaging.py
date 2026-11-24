import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("release_package", Path(__file__).parents[1] / "tools/package.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def test_allowlist_excludes_unlisted_files_and_archive_is_reproducible(tmp_path):
    (tmp_path / "public.py").write_text("print('hello')\n")
    (tmp_path / "private.txt").write_text("unpublished working notes")
    selected = package.collect(tmp_path, ["public.py"])
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    package.write_archive(a, selected, prefix="project/")
    package.write_archive(b, selected, prefix="project/")
    assert a.read_bytes() == b.read_bytes()
    with zipfile.ZipFile(a) as z:
        assert set(z.namelist()) == {"project/public.py", "project/sha256.json"}
        assert set(json.loads(z.read("project/sha256.json"))) == {"public.py"}


def test_sensitive_text_and_symlink_or_traversal_are_rejected(tmp_path):
    path = tmp_path / "public.py"
    path.write_text("endpoint = 'http://" + ".".join(["192", "168", "1", "1"]) + "'\n")
    with pytest.raises(ValueError, match="network"):
        package.collect(tmp_path, [path.name])
    path.write_text("token = '" + "hf_" + "A" * 30 + "'\n")
    with pytest.raises(ValueError, match="credential"):
        package.collect(tmp_path, [path.name])
    path.write_text("safe")
    link = tmp_path / "link.py"
    link.symlink_to(path)
    for names in (["../public.py"], [".env"], ["link.py"]):
        with pytest.raises(ValueError):
            package.collect(tmp_path, names)
