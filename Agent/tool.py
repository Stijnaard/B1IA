"""Ontology Tool: checks a Manchester OWL query against the ontology with HermiT.

Implements Eq. 3 of the report:

    R(Q) = Entailed      if O |= Q(s)
           Contradicted  if O |= not Q(s)
           Unknown       otherwise

The query is parsed with the Manchester syntax parser of owlapy
(https://github.com/dice-group/owlapy) and checked with HermiT through owlapy's
SyncReasoner (OWL API). HermiT is a tableau reasoner, so it computes the negation
normal form and the tableau branches itself.

Subject:
  * an individual (e.g. company:mcdonalds): instance check, as in Table 2.
  * empty: satisfiability check (Scenarios 2 and 5): Contradicted when nothing can be Q.
  * a class S: Entailed when every S is Q, Contradicted when no S can be Q.
"""
from __future__ import annotations

import difflib
import os
import re
import threading
from importlib import resources
from pathlib import Path

import jpype
import jpype.config
from langchain.tools import tool

from configs import BASE_DIR, ONTOLOGY_PATH, ONTOLOGY_IRI, ONTOLOGY_NAMES, JAVA_EXE

ENTAILED = 'Entailed'
CONTRADICTED = 'Contradicted'
UNKNOWN = 'Unknown'
ERROR = 'Error'


# --- Java -----------------------------------------------------------------------------
# owlapy runs the OWL API and HermiT in a Java VM inside this Python process (via JPype).
# We start that VM ourselves, once, on the main thread when this module is imported:
#  * LangGraph runs tool calls in worker threads; letting owlapy start Java there made the
#    process hang when Python exits.
#  * the heap is capped so Java does not compete with the Ollama model for memory.
#  * Java is not shut down at exit (that is the step that hangs); the OS cleans it up.
def _start_java():
    if jpype.isJVMStarted():
        return
    if JAVA_EXE:
        os.environ['JAVA_HOME'] = str(Path(JAVA_EXE).resolve().parent.parent)
    with resources.path('owlapy', 'jar_dependencies') as jar_folder:
        jars = [str(jar) for jar in Path(jar_folder).glob('*.jar')]
    jpype.config.destroy_jvm = False
    try:
        jpype.startJVM('-Xmx1g', '-Dorg.slf4j.simpleLogger.defaultLogLevel=warn', classpath=jars)
    except Exception as e:
        raise RuntimeError('Could not start Java for the HermiT reasoner. Set [REASONER] JAVA in config.ini '
                           'to a java executable, e.g. /Applications/Protégé.app/Contents/jre/bin/java') from e


_start_java()

from owlapy.class_expression import OWLObjectComplementOf, OWLObjectIntersectionOf, OWLClass  # noqa: E402
from owlapy.iri import IRI  # noqa: E402
from owlapy.owl_axiom import OWLClassAssertionAxiom, OWLSubClassOfAxiom  # noqa: E402
from owlapy.owl_individual import OWLNamedIndividual  # noqa: E402
from owlapy.owl_reasoner import SyncReasoner  # noqa: E402
from owlapy.parser import ManchesterOWLSyntaxParser  # noqa: E402

# Individuals are named like "food:burger". The parser reads "food:" as a namespace prefix, so
# each such prefix is mapped to ONTOLOGY_IRI + "food:", which gives back the full IRI.
PREFIXES = {name.split(':')[0]: ONTOLOGY_IRI + name.split(':')[0] + ':'
            for name in ONTOLOGY_NAMES['individual'] if ':' in name}

PARSER = ManchesterOWLSyntaxParser(ONTOLOGY_IRI, prefixes=PREFIXES)
REASONER = SyncReasoner(ontology=str(BASE_DIR / ONTOLOGY_PATH), reasoner='HermiT')
if not REASONER.has_consistent_ontology():
    raise RuntimeError('The base ontology is inconsistent; fix it in Protégé before running the agent.')
_lock = threading.Lock()  # LangGraph runs parallel tool calls in threads; HermiT is not thread-safe


# --- Checking the names in a query ----------------------------------------------------
# The owlapy parser accepts any name: a made-up term silently becomes a new, empty entity,
# which always gives Unknown. So every name is looked up in the ontology first, and the
# QueryAgent gets an error message it can act on.
KEYWORDS = {'and', 'or', 'not', 'some', 'only', 'value', 'min', 'max', 'exactly', 'inverse', 'self', 'that'}
RESTRICTION_KEYWORDS = {'some', 'only', 'value', 'min', 'max', 'exactly', 'self'}
BUILT_INS = {'thing', 'owl:thing', 'nothing', 'owl:nothing', 'true', 'false'}
DATATYPES = {'integer', 'int', 'float', 'double', 'decimal', 'string', 'boolean'}
TOKEN = re.compile(r'"[^"]*"|-?\d+(?:\.\d+)?|[A-Za-z_][\w:\'.\-]*|\S')

KIND = {name: kind for kind, names in ONTOLOGY_NAMES.items() for name in names}
BY_LOWER: dict[str, list[str]] = {}
for _name in KIND:
    BY_LOWER.setdefault(_name.lower(), []).append(_name)


class QueryError(ValueError):
    pass


def resolve(name: str) -> str:
    """Return the ontology name for `name`, forgiving a leading ':' / 'IAOntology:' and capitalisation."""
    candidates = [name, name.lstrip(':')]
    if ':' in name:
        candidates.append(name.split(':', 1)[1])
    for c in candidates:
        if c in KIND:
            return c
    for c in candidates:
        if len(BY_LOWER.get(c.lower(), [])) == 1:
            return BY_LOWER[c.lower()][0]
    close = difflib.get_close_matches(name, list(KIND), n=5, cutoff=0.6)
    hint = f" Close matches: {', '.join(close)}." if close else ''
    raise QueryError(f"Unknown term {name!r}: it does not exist in the ontology.{hint} "
                     "Only use terms from the ontology structure, or label the claim Not-Verifiable.")


def normalise_query(query: str) -> str:
    """Replace every name by its ontology spelling and check that it is used in the right place."""
    tokens = [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(query)]
    out, last, in_braces = [], 0, False
    for i, (tok, start, end) in enumerate(tokens):
        prev = tokens[i - 1][0].lower() if i > 0 else ''
        nxt = tokens[i + 1][0].lower() if i + 1 < len(tokens) else ''
        if tok.lower() in ('and', 'or', 'that') and nxt in ('and', 'or', 'that', ')', '}', ''):
            # the parser would read a second 'and' as a class called "and"
            raise QueryError(f"Invalid Manchester syntax in {query!r}: '{tok}' must be followed by a class expression.")
        if tok == '{':
            in_braces = True
        elif tok == '}':
            in_braces = False
        is_datatype = tok.lower().startswith('xsd:') or tok.lower() in DATATYPES
        is_name = re.match(r'[A-Za-z_]', tok) and tok.lower() not in KEYWORDS | BUILT_INS and not is_datatype
        if is_name:
            name = resolve(tok)
            kind = KIND[name]
            if kind == 'individual' and prev != 'value' and not in_braces:
                raise QueryError(f"{name!r} is an individual, not a class. Pass it as the subject, "
                                 f"use {{{name}}}, or use 'property value {name}'.")
            if kind != 'individual' and (prev == 'value' or in_braces):
                raise QueryError(f"After 'value' or inside {{...}} an individual is needed, but {name!r} is a {kind.replace('_', ' ')}.")
            if kind.endswith('property') and nxt not in RESTRICTION_KEYWORDS:
                raise QueryError(f"Property {name!r} must be followed by some/only/value/min/max/exactly.")
            out.append(query[last:start] + name)
            last = end
    out.append(query[last:])
    return ''.join(out)


# --- The check (Eq. 3) ----------------------------------------------------------------
def evaluate_query(query: str, subject: str = '') -> str:
    """Return ENTAILED, CONTRADICTED or UNKNOWN; raises QueryError for an invalid query."""
    query = normalise_query(query.strip().strip('`'))
    subject = subject.strip().strip('`')
    subject = resolve(subject) if subject else ''
    if subject and KIND[subject].endswith('property'):
        raise QueryError(f"The subject {subject!r} must be an individual (or a class), not a property.")
    try:
        expression = PARSER.parse_expression(query)
    except Exception as e:
        raise QueryError(f"Invalid Manchester syntax in {query!r}: {str(e).splitlines()[0]}") from e
    negation = OWLObjectComplementOf(expression)

    with _lock:
        if subject and KIND[subject] == 'individual':        # instance check
            individual = OWLNamedIndividual(IRI.create(ONTOLOGY_IRI + subject))
            if REASONER.is_entailed(OWLClassAssertionAxiom(individual, expression)):
                return ENTAILED
            if REASONER.is_entailed(OWLClassAssertionAxiom(individual, negation)):
                return CONTRADICTED
        elif subject:                                         # class subject: every S is Q?
            subject_class = OWLClass(IRI.create(ONTOLOGY_IRI + subject))
            if REASONER.is_entailed(OWLSubClassOfAxiom(subject_class, expression)):
                return ENTAILED
            if not REASONER.is_satisfiable(OWLObjectIntersectionOf((subject_class, expression))):
                return CONTRADICTED
        else:                                                 # no subject: can anything be Q?
            if not REASONER.is_satisfiable(expression):
                return CONTRADICTED
            if not REASONER.is_satisfiable(negation):
                return ENTAILED
    return UNKNOWN


@tool
def check_ontology(query: str, subject: str = '', sentence: int = 0, claim: str = '') -> str:
    """Check one claim against the ontology with the HermiT reasoner.

    Args:
        query: class expression C in Manchester OWL Syntax, e.g. "activeIn value country:netherlands".
            Write C itself, never its negation; the tool also checks the complement.
        subject: the individual the claim is about, e.g. "company:mcdonalds". Leave empty to check
            whether anything can satisfy C at all (e.g. "CoeliacPerson and canEat value food:burger").
        sentence: number of the summary sentence the claim comes from (starting at 1).
        claim: the claim in plain words, as extracted from the sentence.

    Returns "Entailed" (the ontology entails the claim), "Contradicted" (the ontology entails the
    opposite) or "Unknown" (neither, open world assumption), or "Error: ..." when the query is
    invalid; in that case fix the query and call the tool again.
    """
    print(f'checking query: subject={subject!r} query={query!r}', flush=True)
    try:
        result = evaluate_query(query, subject)
    except QueryError as e:
        result = f'{ERROR}: {e}'
    except Exception as e:  # never crash the agent; report the problem so it can retry
        result = f'{ERROR}: {type(e).__name__}: {e}'
    print(f'  -> {result}', flush=True)
    return result
