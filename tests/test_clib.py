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
