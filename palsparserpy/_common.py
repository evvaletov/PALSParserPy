"""
Helpers the three translators share: name and parameter patterns, number
handling, derived values, PALS name/value lists, facility lookup, and the
multipole representations.
"""

from __future__ import annotations

import cmath
import math
import re
from typing import Dict, List, Optional, Tuple

from .node import YAMLNode

__all__ = ["NAME_RE", "MULTIPOLE_RE",
           "approx", "fmt", "try_float", "fold", "times", "scale", "shift",
           "name_value_pairs", "value_text",
           "ctrl_variables", "facility_entry", "facility_props",
           "FullRepresentation", "ABRepresentation", "tilt_rotation",
           "fill_multipoles"]

# A plain identifier: what every target format accepts as a bare name, and so
# what a translated element, attribute or variable name has to look like.
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# A PALS path naming one multipole component, as `MagneticMultipoleP.Kn1L`.
# The groups are: K or B (normalized or field), n or s (normal or skew), the
# multipole order, and a trailing L when the component is length integrated.
MULTIPOLE_RE = re.compile(r"^MagneticMultipoleP\.([KB])([ns])([0-9]+)(L?)$")

# The tolerance Julia's `isapprox` uses by default, which is what the reference
# implementation of these translators compared with.
_RTOL = math.sqrt(2.0 ** -52)


def approx(a: float, b: float, rtol: float = _RTOL, atol: float = 0.0) -> bool:
    """Whether two numbers agree to a relative tolerance.

    Note that ``approx(x, 0)`` is exactly ``x == 0``: with no absolute tolerance
    there is nothing for a relative one to be relative to. That is deliberate --
    a strength of 1e-30 is a strength that was written down, and the translators
    that ask this question mean "was anything stated here at all".
    """
    return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))


def fmt(value) -> str:
    """Render a number for a lattice file.

    Python's own ``repr`` is the shortest text that reads back as the same float,
    which is what a lattice file wants; only the exponent form is adjusted, from
    ``1e-05`` to the ``1.0e-5`` the accelerator formats are written with.
    """
    if isinstance(value, bool) or isinstance(value, int):
        return str(value)
    text = repr(float(value))
    if "e" in text:
        mantissa, _, exponent = text.partition("e")
        if "." not in mantissa:
            mantissa += ".0"
        text = f"{mantissa}e{int(exponent)}"
    return text


def try_float(text) -> Optional[float]:
    """The number ``text`` spells, or ``None`` if it does not spell one.

    A PALS parameter may be written as an expression, which only the target
    program can evaluate, or as a plain number, which the translation can work
    with; this is what tells the two apart.
    """
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


# ─── derived values ──────────────────────────────────────────────────────────
# A target value is often not a PALS parameter as written but something computed
# from one or more of them. Where every input is a number the computation can be
# done here; where any is an expression it has to be handed to the target program
# to evaluate, written out in the target's own syntax. These four take care of
# that split, so that each translator writes only the expression text.

def fold(text: str, f, *args: str) -> str:
    """``text``, or the number it comes to when every one of ``args`` is a number.

    A PALS parameter may be written as an expression, which only the target
    program can evaluate, or as a plain number, which the translation can work
    with. Where a target value has to be derived from several PALS ones, ``text``
    is that derivation written as a target expression and ``f`` is the same
    derivation as a function, applied here when all of its inputs parse.
    """
    values = [try_float(a) for a in args]
    if any(v is None for v in values):
        return text
    result = f(*values)
    return fmt(result) if math.isfinite(result) else text


def times(a: str, b: str) -> str:
    """The product of two target values, without the clutter of a factor of one.

    Two numbers are multiplied out here; a unit factor -- which is what an element
    of unit length gives, and what several of the bend derivations reduce to --
    comes back as the other operand alone rather than as a product with nothing in
    it.
    """
    va, vb = try_float(a), try_float(b)
    if va is not None and vb is not None:
        return fmt(va * vb)
    if va == 1:
        return b
    if vb == 1:
        return a
    return f"({a}) * ({b})"


def scale(text: str, factor: float) -> str:
    """``text`` scaled by ``factor``, as a target value.

    PALS and the target formats differ in the units of nearly every quantity that
    is not a length or an angle: energies are eV against GeV, voltages V against
    MV, frequencies Hz against MHz. A value written as a number is scaled here and
    comes out a number; one written as an expression -- a constant, say -- is left
    for the target program to evaluate and comes out an expression.
    """
    if factor == 1:
        return text
    value = try_float(text)
    return f"({text}) * {fmt(factor)}" if value is None else fmt(value * factor)


def shift(text: str, offset: float) -> str:
    """``text`` with ``offset`` added, as a target value. As with :func:`scale`, a
    number comes out a number and an expression comes out an expression."""
    if offset == 0:
        return text
    value = try_float(text)
    return f"({text}) + {fmt(offset)}" if value is None else fmt(value + offset)


# ─── PALS name/value lists ───────────────────────────────────────────────────

def value_text(node: YAMLNode) -> str:
    """The text of a value ``node``, with a value left unwritten taken as PALS'
    default of zero."""
    text = node.value.strip()
    return "0" if text in ("", "~", "null") else text


def name_value_pairs(node: YAMLNode) -> List[Tuple[str, str]]:
    """A PALS name/value list as ``(name, value-text)`` pairs, in definition
    order.

    Accepts both forms the standard allows for such a list: a map (``vv: 0.3``)
    and a sequence of single-key maps (``- vv: 0.3``). An entry written with no
    value takes PALS' default of zero, and one whose value is a structure rather
    than a single value is skipped.
    """
    pairs = []

    def add(child):
        if child.is_map() or child.is_sequence():
            return
        pairs.append((child.node_key(), value_text(child)))

    if node.is_map():
        for key in node.keys():
            add(node[key])
    elif node.is_sequence():
        for entry in node:
            for i in range(len(entry)):
                add(entry.child(i))
    return pairs


def ctrl_variables(props: YAMLNode) -> List[Tuple[str, str]]:
    """A controller's ``variables`` as ``(name, value-text)`` pairs, in definition
    order."""
    if "variables" not in props:
        return []
    return name_value_pairs(props["variables"])


# ─── facility lookup ─────────────────────────────────────────────────────────

def facility_entry(facility: YAMLNode, name: str) -> Optional[YAMLNode]:
    """The ``facility`` entry named ``name``, or ``None`` if there is none.

    The entry is the single-key map the translators take as an element;
    :func:`facility_props` gives its properties.
    """
    for ele in facility:
        if ele.child(0).node_key() == name:
            return ele
    return None


def facility_props(facility: YAMLNode, name: str) -> Optional[YAMLNode]:
    """The property map of the ``facility`` entry named ``name``, or ``None`` if
    there is none."""
    ele = facility_entry(facility, name)
    return None if ele is None else ele.child(0)


# ─── multipole representations ───────────────────────────────────────────────

class FullRepresentation:
    """Raw, over-parametrized multipole form filled directly from PALS-YAML.

    Holds, keyed by multipole order, whether each coefficient is ``normalized``
    (K vs. B) and ``integrated`` (field integral vs. field strength), its
    ``magnitude`` (a ``[normal, skew]`` pair), and its ``tilt``, together with the
    element length ``L``. It is down-converted to whichever element-specific
    representation the element kind requires.
    """

    __slots__ = ("normalized", "integrated", "magnitude", "tilt", "L")

    def __init__(self):
        self.normalized: Dict[int, bool] = {}
        self.integrated: Dict[int, bool] = {}
        self.magnitude: Dict[int, List[float]] = {}
        self.tilt: Dict[int, float] = {}
        self.L: float = 1.0


class ABRepresentation:
    """Element-specific multipole form: only the final A/B field integrals.

    ``A`` and ``B`` map each multipole order to its skew and normal field
    integral, respectively. Built from a :class:`FullRepresentation` by combining
    each multipole's magnitude, length and tilt into a complex field integral and
    storing its imaginary/real parts.
    """

    __slots__ = ("A", "B")

    def __init__(self, full: FullRepresentation):
        self.A: Dict[int, float] = {}
        self.B: Dict[int, float] = {}
        for order in sorted(full.magnitude):
            length = 1.0 if full.integrated[order] else full.L
            tilt = full.tilt.get(order, 0.0)
            fact = 1 / math.factorial(order)
            b_ia = ((fact * length) * complex(*full.magnitude[order])
                    * tilt_rotation(order, tilt))
            self.A[order] = b_ia.imag
            self.B[order] = b_ia.real


def tilt_rotation(order: int, tilt: float) -> complex:
    """The factor that rotates an ``order`` multipole of the given ``tilt`` into
    normal and skew parts.

    A tilt of ``T`` rotates an order-``N`` field by ``(N+1) * T`` in the
    normal/skew plane: both PALS and Bmad write the field as
    ``(1/N!) (normal + i * skew) exp(-i (N+1) T)``.
    """
    return cmath.exp(-1j * (order + 1) * tilt)


def fill_multipoles(full: FullRepresentation, mmP: YAMLNode,
                    name: str) -> FullRepresentation:
    """Populate ``full`` from a PALS ``MagneticMultipoleP`` map.

    Parse each key of ``mmP`` into a multipole order and store its magnitude,
    ``normalized``, ``integrated`` and ``tilt`` attributes in ``full``; ``name``
    is used in error messages. Returns ``full``.
    """
    for key in mmP.keys():
        order = int("".join(ch for ch in key if ch.isdigit()))
        if key.startswith("tilt"):
            if order in full.tilt:
                raise ValueError(f"{name} conflicting multipole definitions {key}")
            full.tilt[order] = mmP[key].as_float()
        else:
            component = 0 if key[1] == "n" else 1
            if order not in full.magnitude:
                full.integrated[order] = key.endswith("L")
                full.normalized[order] = key.startswith("K")
                full.magnitude[order] = [0.0, 0.0]
            full.magnitude[order][component] = mmP[key].as_float()
    return full
