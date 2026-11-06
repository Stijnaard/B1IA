"""Ontology Tool: checks a Manchester OWL query against the ontology with HermiT.

Implements Eq. 3 of the report:

    R(Q) = Entailed      if O |= Q(s)
           Contradicted  if O |= not Q(s)
           Unknown       otherwise

The query is parsed with the Manchester syntax parser of owlapy
(https://github.com/dice-group/owlapy) and checked with HermiT through owlapy's
SyncReasoner (OWL API). HermiT is a tableau reasoner: it computes the negation
normal form and the tableau branches itself.

Subject:
  * an individual (e.g. company:mcdonalds): instance check, as in Table 2.
  * empty: satisfiability check (Scenarios 2 and 5): Contradicted when nothing
    can be an instance of Q.
"""
from __future__ import annotations

import difflib
import os
import re
import threading
from pathlib import Path

from langchain.tools import tool

from configs import BASE_DIR, ONTOLOGY_PATH, JAVA_EXE, ONTOLOGY_NAMES, INDIVIDUAL_NAMES

# owlapy starts a Java VM through JPype, which finds Java via JAVA_HOME
if JAVA_EXE:
    os.environ['JAVA_HOME'] = str(Path(JAVA_EXE).resolve().parent.parent)
# hide the OWL API's INFO messages about every parsed axiom
os.environ.setdefault('JAVA_TOOL_OPTIONS', '-Dorg.slf4j.simpleLogger.defaultLogLevel=warn')

import jpype.config
# Java is started from a LangGraph worker thread; shutting it down at exit then hangs forever,
# so let the operating system clean it up when Python exits.
jpype.config.destroy_jvm = False

from owlapy.class_expression import OWLObjectComplementOf
from owlapy.iri import IRI
from owlapy.owl_axiom import OWLClassAssertionAxiom
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_reasoner import SyncReasoner
from owlapy.parser import ManchesterOWLSyntaxParser

ENTAILED = 'Entailed'
CONTRADICTED = 'Contradicted'
UNKNOWN = 'Unknown'
ERROR = 'Error'

ONTOLOGY_FILE = BASE_DIR / ONTOLOGY_PATH
NAMESPACE = 'http://www.semanticweb.org/stijnroos/ontologies/2026/8/IAOntology#'

# Individuals are named like "food:burger". The parser reads "food:" as a prefix, so every such
# prefix is mapped to NAMESPACE + "food:", which gives back the full IRI of the individual.
PREFIXES = {name.split(':')[0]: NAMESPACE + name.split(':')[0] + ':'
            for name in INDIVIDUAL_NAMES if ':' in name}

MANCHESTER_KEYWORDS = {'and', 'or', 'not', 'some', 'only', 'value', 'min', 'max', 'exactly',
                       'inverse', 'self', 'that', 'thing', 'nothing', 'true', 'false',
                       'integer', 'int', 'float', 'double', 'decimal', 'string', 'boolean'}

_parser = ManchesterOWLSyntaxParser(NAMESPACE, prefixes=PREFIXES)
_reasoner = None
_lock = threading.Lock()  # tool calls run in parallel threads; HermiT is not thread-safe


def get_reasoner() -> SyncReasoner:
    """Load the ontology and HermiT once, and check that the base ontology is consistent."""
    global _reasoner
    if _reasoner is None:
        _reasoner = SyncReasoner(ontology=str(ONTOLOGY_FILE), reasoner='HermiT')
        if not _reasoner.has_consistent_ontology():
            raise RuntimeError('The base ontology is inconsistent; fix it in Protege before running the agent.')
    return _reasoner


NAME_PATTERN = re.compile(r'"[^"]*"|[A-Za-z_][\w:\'.\-]*')
NAMES_BY_LOWER = {name.lower(): name for name in ONTOLOGY_NAMES}


def fix_case(word: str) -> str:
    """Return the ontology spelling of a name that only differs in upper/lower case."""
    if word in ONTOLOGY_NAMES:
        return word
    return NAMES_BY_LOWER.get(word.lower(), word)


def suggestions(word: str) -> list[str]:
    """Ontology names close to an unknown word, also matching on the part after the colon,
    so that e.g. 'streamingServices:netflix' suggests 'company:netflix'."""
    local = word.split(':')[-1].lower()
    same_local = [n for n in ONTOLOGY_NAMES if n.split(':')[-1].lower() == local]
    close = difflib.get_close_matches(word, ONTOLOGY_NAMES, 3)
    return list(dict.fromkeys(same_local + close))[:4]


def check_terms(text: str) -> tuple[str, list[str]]:
    """Fix the capitalisation of known names and list the names that do not exist in the ontology.

    The parser accepts any name and would treat a made-up term as a new, empty entity,
    which always gives Unknown. Checking the names first lets the QueryAgent correct them.
    """
    unknown = []

    def replace(match):
        word = match.group()
        if word.startswith('"') or word.lower() in MANCHESTER_KEYWORDS or word.startswith('xsd:'):
            return word
        fixed = fix_case(word)
        if fixed not in ONTOLOGY_NAMES:
            unknown.append(word)
        return fixed

    return NAME_PATTERN.sub(replace, text), unknown


def evaluate_query(query: str, subject: str = '') -> str:
    """Return ENTAILED, CONTRADICTED or UNKNOWN, or an 'Error: ...' message for invalid input."""
    query, subject = query.strip().strip('`'), (subject or '').strip().strip('`')

    query, unknown_query = check_terms(query)
    subject, unknown_subject = check_terms(subject)
    unknown = unknown_query + unknown_subject
    if unknown:
        hints = []
        for w in unknown:
            close = suggestions(w)
            hints.append(f"{w} (did you mean: {', '.join(close)})" if close else w)
        return (f"{ERROR}: these terms do not exist in the ontology: {'; '.join(hints)}. "
                "Only use terms from the ontology structure, or label the claim Not-Verifiable.")
    if subject and subject not in INDIVIDUAL_NAMES:
        return f"{ERROR}: the subject {subject!r} is not an individual. Use an individual or leave the subject empty."
    try:
        expression = _parser.parse_expression(query)
    except Exception as e:
        return f"{ERROR}: invalid Manchester syntax in {query!r}: {str(e).splitlines()[0]}"

    with _lock:
        reasoner = get_reasoner()
        if subject:
            individual = OWLNamedIndividual(IRI.create(NAMESPACE + subject))
            if reasoner.is_entailed(OWLClassAssertionAxiom(individual, expression)):
                return ENTAILED
            if reasoner.is_entailed(OWLClassAssertionAxiom(individual, OWLObjectComplementOf(expression))):
                return CONTRADICTED
        else:
            if not reasoner.is_satisfiable(expression):
                return CONTRADICTED   # nothing can be an instance of Q
            if not reasoner.is_satisfiable(OWLObjectComplementOf(expression)):
                return ENTAILED       # everything is an instance of Q
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
    result = evaluate_query(query, subject)
    print(f'  -> {result}', flush=True)
    return result
