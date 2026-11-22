"""Where PALSParserPy looks for the PALSParserCpp shared library."""

import os
import sys

import pytest

from palsparserpy import _clib


@pytest.fixture
def no_overrides(monkeypatch):
    monkeypatch.delenv("PALS_PARSER_CPP_LIB", raising=False)
    monkeypatch.delenv("PALS_PARSER_CPP_DIR", raising=False)


def _env_libdir(prefix):
    if sys.platform == "win32":
        return os.path.join(prefix, "Library", "bin")
    return os.path.join(prefix, "lib")


def test_environment_library_dir_is_searched_last(no_overrides, monkeypatch,
                                                  tmp_path):
    # A conda package installs the library into the environment's own library
    # directory; a developer's checkout build must still win over it.
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    ext = _clib._dlext()
    libdir = _env_libdir(str(tmp_path))
    expected = [os.path.join(libdir, f"libPALSParserCpp.{ext}"),
                os.path.join(libdir, f"PALSParserCpp.{ext}")]

    candidates = _clib._candidates()

    assert candidates[-2:] == expected
    assert all(os.path.dirname(c) != libdir for c in candidates[:-2])


def test_explicit_library_path_is_searched_first(no_overrides, monkeypatch):
    monkeypatch.setenv("PALS_PARSER_CPP_LIB", "/somewhere/libPALSParserCpp.so")
    assert _clib._candidates()[0] == "/somewhere/libPALSParserCpp.so"


# ─── the library's version ───────────────────────────────────────────────────

class _FakeLib:
    """Stands in for a loaded library that reports ``version`` (bytes), or has
    no pals_parser_cpp_version at all when ``version`` is None."""

    def __init__(self, version):
        if version is not None:
            def pals_parser_cpp_version():
                return version
            self.pals_parser_cpp_version = pals_parser_cpp_version


def _text(v):
    return ".".join(map(str, v)).encode("ascii")


def test_library_in_use_is_a_supported_version():
    lo, hi = _clib.SUPPORTED_LIBRARY_VERSIONS
    version = tuple(int(p) for p in _clib.library_version().split("."))
    assert lo <= version <= hi


@pytest.mark.parametrize("which", [0, 1])
def test_supported_versions_pass(which):
    v = _clib.SUPPORTED_LIBRARY_VERSIONS[which]
    _clib._check_version(_FakeLib(_text(v)), "/lib/libPALSParserCpp.so")


def test_library_without_a_version_function_is_refused():
    with pytest.raises(RuntimeError, match=r"no pals_parser_cpp_version\(\)"):
        _clib._check_version(_FakeLib(None), "/lib/libPALSParserCpp.so")


@pytest.mark.parametrize("version", [b"0.0.0", b"999.0.0"])
def test_version_outside_the_range_is_refused(version):
    with pytest.raises(RuntimeError, match=version.decode()):
        _clib._check_version(_FakeLib(version), "/lib/libPALSParserCpp.so")


@pytest.mark.parametrize("version", [b"", b"0.0", b"0.0.3.1", b"0.x.3"])
def test_unreadable_version_is_refused(version):
    with pytest.raises(RuntimeError, match="not major.minor.patch"):
        _clib._check_version(_FakeLib(version), "/lib/libPALSParserCpp.so")
