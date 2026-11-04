"""Small parser for Manchester OWL Syntax class expressions -> owlready2 constructs.

Replaces the TableauAgent and SparQLAgent LLM nodes (see README section
"Deviation from the report"). The parser covers the subset of Manchester
syntax the QueryAgent needs for the ontology:

    C or D          C and D (also: C that D)      not C
    p some C        p only C        p value a      p Self
    p min n [C]     p max n [C]     p exactly n [C]
    inverse p some C
    {a, b, c}       (C)             Thing / Nothing
    data ranges:    hasAge value 16      hasAge some integer[>= 18, <= 110]
                    hasAge some [< 18]   isPublicOwned value true

Names are resolved against the loaded ontology. Unknown names raise a
ManchesterError that lists close matches, so the QueryAgent can retry.
"""
from __future__ import annotations

import datetime
import difflib
import re

from owlready2 import (
    And, ConstrainedDatatype, DataPropertyClass, Inverse, Not, Nothing,
    ObjectPropertyClass, OneOf, Or, Thing, ThingClass,
)


class ManchesterError(ValueError):
    """Raised when a query cannot be parsed or refers to unknown terms."""


KEYWORDS = {'and', 'or', 'not', 'some', 'only', 'value', 'min', 'max',
            'exactly', 'inverse', 'self', 'that'}

DATATYPES = {
    'integer': int, 'int': int, 'nonnegativeinteger': int, 'positiveinteger': int,
    'long': int, 'short': int,
    'float': float, 'double': float, 'decimal': float, 'real': float,
    'string': str, 'boolean': bool, 'bool': bool,
    'datetime': datetime.datetime, 'date': datetime.date,
}

FACETS = {'>=': 'min_inclusive', '>': 'min_exclusive',
          '<=': 'max_inclusive', '<': 'max_exclusive'}

TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<iri><[^>\s]+>)
  | (?P<string>"[^"]*"(?:\^\^[\w:]+)?)
  | (?P<number>-?\d+(?:\.\d+)?(?![\w:]))
  | (?P<facet>>=|<=|>|<)
  | (?P<punct>[(){}\[\],])
  | (?P<name>[A-Za-z_][\w:.\-']*)
""", re.VERBOSE)


def tokenize(text: str) -> list[str]:
    # tolerate typographic quotes and stray backticks produced by LLMs
    text = text.replace('“', '"').replace('”', '"').replace('`', '').strip()
    tokens, pos = [], 0
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if m is None:
            raise ManchesterError(f"Unexpected character {text[pos]!r} at position {pos} in {text!r}")
        if m.lastgroup != 'ws':
            tokens.append(m.group())
        pos = m.end()
    return tokens


class Resolver:
    """Looks up ontology entities by (local) name."""

    def __init__(self, onto):
        self.onto = onto
        world = onto.world
        entities = (list(world.classes()) + list(world.object_properties())
                    + list(world.data_properties()) + list(world.individuals()))
        self.by_name = {e.name: e for e in entities}
        self.by_lower: dict[str, list] = {}
        for e in entities:
            self.by_lower.setdefault(e.name.lower(), []).append(e)

    def lookup(self, name: str):
        if name.startswith('<') and name.endswith('>'):
            entity = self.onto.world[name[1:-1]]
            if entity is None:
                raise ManchesterError(f"Unknown IRI {name}")
            return entity
        if name in ('Thing', 'owl:Thing'):
            return Thing
        if name in ('Nothing', 'owl:Nothing'):
            return Nothing
        candidates = [name, name.lstrip(':')]
        if ':' in name:  # tolerate a namespace prefix such as "IAOntology:" or ":"
            candidates.append(name.split(':', 1)[1])
        for c in candidates:
            if c in self.by_name:
                return self.by_name[c]
        for c in candidates:
            matches = self.by_lower.get(c.lower(), [])
            if len(matches) == 1:
                return matches[0]
        close = difflib.get_close_matches(name, list(self.by_name), n=5, cutoff=0.6)
        hint = f" Close matches: {', '.join(close)}." if close else ""
        raise ManchesterError(
            f"Unknown term {name!r}: it does not exist in the ontology.{hint} "
            "Only use terms from the ontology structure, or label the claim Not-Verifiable.")

    def kind(self, entity) -> str:
        if entity is Thing or entity is Nothing or isinstance(entity, ThingClass):
            return 'class'
        if isinstance(entity, ObjectPropertyClass):
            return 'object_property'
        if isinstance(entity, DataPropertyClass):
            return 'data_property'
        if isinstance(entity, Thing):
            return 'individual'
        return 'other'


class Parser:
    def __init__(self, text: str, onto):
        self.text = text
        self.tokens = tokenize(text)
        self.i = 0
        self.r = Resolver(onto)

    # -- token helpers -------------------------------------------------
    def peek(self, offset: int = 0) -> str | None:
        j = self.i + offset
        return self.tokens[j] if j < len(self.tokens) else None

    def peek_kw(self) -> str | None:
        t = self.peek()
        return t.lower() if t is not None and t.lower() in KEYWORDS else None

    def next(self) -> str:
        t = self.peek()
        if t is None:
            raise ManchesterError(f"Unexpected end of query {self.text!r}")
        self.i += 1
        return t

    def expect(self, tok: str):
        t = self.next()
        if t.lower() != tok:
            raise ManchesterError(f"Expected {tok!r} but found {t!r} in {self.text!r}")

    # -- grammar -------------------------------------------------------
    def parse(self):
        if not self.tokens:
            raise ManchesterError("Empty query")
        expr = self.parse_or()
        if self.peek() is not None:
            raise ManchesterError(f"Unexpected token {self.peek()!r} in {self.text!r}")
        return expr

    def parse_or(self):
        parts = [self.parse_and()]
        while self.peek_kw() == 'or':
            self.next()
            parts.append(self.parse_and())
        return parts[0] if len(parts) == 1 else Or(parts)

    def parse_and(self):
        parts = [self.parse_not()]
        while self.peek_kw() in ('and', 'that'):
            self.next()
            parts.append(self.parse_not())
        return parts[0] if len(parts) == 1 else And(parts)

    def parse_not(self):
        if self.peek_kw() == 'not':
            self.next()
            return Not(self.parse_not())
        return self.parse_restriction_or_atomic()

    def parse_restriction_or_atomic(self):
        tok = self.peek()
        if tok is None:
            raise ManchesterError(f"Unexpected end of query {self.text!r}")
        if tok == '(':
            self.next()
            expr = self.parse_or()
            self.expect(')')
            return expr
        if tok == '{':
            return self.parse_one_of()
        if tok.lower() == 'inverse':
            self.next()
            prop = self.r.lookup(self.next())
            if self.r.kind(prop) != 'object_property':
                raise ManchesterError(f"'inverse' needs an object property, got {prop.name!r}")
            return self.parse_object_restriction(Inverse(prop), prop.name)
        if tok.lower() in KEYWORDS or tok in ',)]}' or tok in FACETS:
            raise ManchesterError(f"Unexpected token {tok!r} in {self.text!r}")

        entity = self.r.lookup(self.next())
        kind = self.r.kind(entity)
        if kind == 'class':
            return entity
        if kind == 'object_property':
            return self.parse_object_restriction(entity, entity.name)
        if kind == 'data_property':
            return self.parse_data_restriction(entity)
        if kind == 'individual':
            raise ManchesterError(
                f"{entity.name!r} is an individual, not a class. Pass it as the subject, "
                f"use {{{entity.name}}}, or use 'property value {entity.name}'.")
        raise ManchesterError(f"Cannot use {entity!r} in a class expression")

    def parse_one_of(self):
        self.expect('{')
        items = []
        while True:
            ent = self.r.lookup(self.next())
            if self.r.kind(ent) != 'individual':
                raise ManchesterError(f"{{...}} may only contain individuals, got {ent.name!r}")
            items.append(ent)
            t = self.next()
            if t == '}':
                break
            if t != ',':
                raise ManchesterError(f"Expected ',' or '}}' but found {t!r} in {self.text!r}")
        return OneOf(items)

    def parse_cardinality(self):
        t = self.next()
        if not re.fullmatch(r'\d+', t):
            raise ManchesterError(f"Expected a number after min/max/exactly, found {t!r}")
        return int(t)

    def has_filler(self) -> bool:
        t = self.peek()
        return t is not None and t not in (')', ',', '}') and self.peek_kw() not in ('and', 'or', 'that')

    def parse_object_restriction(self, prop, name):
        kw = self.peek_kw()
        if kw is None:
            raise ManchesterError(
                f"Property {name!r} must be followed by some/only/value/min/max/exactly in {self.text!r}")
        self.next()
        if kw == 'some':
            return prop.some(self.parse_not())
        if kw == 'only':
            return prop.only(self.parse_not())
        if kw == 'value':
            ind = self.r.lookup(self.next())
            if self.r.kind(ind) != 'individual':
                raise ManchesterError(f"'{name} value' needs an individual, got {ind.name!r}")
            return prop.value(ind)
        if kw == 'self':
            return prop.has_self()
        if kw in ('min', 'max', 'exactly'):
            n = self.parse_cardinality()
            filler = self.parse_not() if self.has_filler() else Thing
            return getattr(prop, kw)(n, filler)
        raise ManchesterError(f"Unexpected {kw!r} after property {name!r}")

    def parse_data_restriction(self, prop):
        kw = self.peek_kw()
        if kw is None:
            raise ManchesterError(
                f"Data property {prop.name!r} must be followed by some/only/value/min/max/exactly")
        self.next()
        if kw == 'value':
            return prop.value(self.parse_literal())
        if kw in ('some', 'only'):
            return getattr(prop, kw)(self.parse_data_range(prop))
        if kw in ('min', 'max', 'exactly'):
            n = self.parse_cardinality()
            rng = self.parse_data_range(prop) if self.has_filler() else None
            return getattr(prop, kw)(n, rng) if rng is not None else getattr(prop, kw)(n)
        raise ManchesterError(f"Unexpected {kw!r} after data property {prop.name!r}")

    def parse_literal(self):
        t = self.next()
        if t.startswith('"'):
            return t.split('"')[1]
        if re.fullmatch(r'-?\d+', t):
            return int(t)
        if re.fullmatch(r'-?\d+\.\d+', t):
            return float(t)
        if t.lower() in ('true', 'false'):
            return t.lower() == 'true'
        raise ManchesterError(f"Expected a literal value, found {t!r}")

    def default_datatype(self, prop):
        for r in getattr(prop, 'range', []) or []:
            if r in (int, float, str, bool):
                return r
        return int

    def parse_data_range(self, prop):
        t = self.peek()
        if t == '{':  # {1, 2, 3}
            self.next()
            values = []
            while True:
                values.append(self.parse_literal())
                t = self.next()
                if t == '}':
                    return OneOf(values)
                if t != ',':
                    raise ManchesterError(f"Expected ',' or '}}' in literal list, found {t!r}")
        datatype = None
        if t is not None and t != '[':
            name = t.split(':')[-1].lower()
            if name not in DATATYPES:
                raise ManchesterError(f"Unknown datatype {t!r}; use e.g. integer, float, string, boolean")
            self.next()
            datatype = DATATYPES[name]
        if self.peek() != '[':
            return datatype or self.default_datatype(prop)
        self.next()
        facets = {}
        while True:
            op = self.next()
            if op not in FACETS:
                raise ManchesterError(f"Expected a facet (>=, >, <=, <) but found {op!r}")
            facets[FACETS[op]] = self.parse_literal()
            t = self.next()
            if t == ']':
                break
            if t != ',':
                raise ManchesterError(f"Expected ',' or ']' in data range, found {t!r}")
        return ConstrainedDatatype(datatype or self.default_datatype(prop), **facets)


def parse_class_expression(text: str, onto):
    """Parse a Manchester class expression into an owlready2 construct bound to `onto`'s world."""
    return Parser(text, onto).parse()


def resolve_subject(name: str, onto):
    """Return (entity, kind) for a subject name; kind is 'individual' or 'class'."""
    r = Resolver(onto)
    entity = r.lookup(name.strip().strip('`'))
    kind = r.kind(entity)
    if kind not in ('individual', 'class'):
        raise ManchesterError(f"The subject {name!r} must be an individual (or a class), not a property")
    return entity, kind
