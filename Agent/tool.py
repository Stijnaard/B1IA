"""Ontology Tool: checks a Manchester OWL query against the ontology with HermiT.

Implements Eq. 3 of the report:

    R(Q) = Entailed      if O |= Q(s)
           Contradicted  if O |= not Q(s)
           Unknown       otherwise

O |= Q(s) holds exactly when O + {(not Q)(s)} is inconsistent, and
O |= (not Q)(s) holds exactly when O + {Q(s)} is inconsistent. The tool therefore
asserts the query and its complement for the subject s in two isolated copies of
the ontology and lets HermiT check each copy for consistency. HermiT performs the
tableau expansion (including the negation normal form) itself, so the query no
longer has to pass through LLM nodes for the NNF and SPARQL translation.

Subjects:
  * an individual (e.g. company:mcdonalds): instance check, as in Table 2.
  * no subject: a fresh anonymous individual is used, i.e. a satisfiability check
    (Scenarios 2 and 5 in the report): Contradicted when Q is unsatisfiable.
  * a class S: subsumption check, Entailed when S is a subclass of Q and
    Contradicted when S and Q are disjoint.
"""
from __future__ import annotations

import uuid
from functools import lru_cache

import owlready2
from langchain.tools import tool
from owlready2 import World, OwlReadyInconsistentOntologyError, Not, Thing, sync_reasoner

from configs import ONTOLOGY_PATH, BASE_DIR, JAVA_EXE
from manchester import ManchesterError, parse_class_expression, resolve_subject

if JAVA_EXE:
    owlready2.JAVA_EXE = JAVA_EXE

ENTAILED = 'Entailed'
CONTRADICTED = 'Contradicted'
UNKNOWN = 'Unknown'
ERROR = 'Error'


def _load_world():
    world = World()
    onto = world.get_ontology(str(BASE_DIR / ONTOLOGY_PATH)).load()
    return world, onto


def _is_consistent(world) -> bool:
    try:
        sync_reasoner(world, infer_property_values=False, debug=0)
    except OwlReadyInconsistentOntologyError:
        return False
    return True


@lru_cache(maxsize=1)
def _check_base_ontology():
    world, _ = _load_world()
    if not _is_consistent(world):
        raise RuntimeError('The base ontology is inconsistent; fix it in Protege before running the agent.')
    return True


def _consistent_with(query: str, subject: str, negate: bool) -> bool:
    """Assert (not) query for the subject in a fresh copy of the ontology and check consistency."""
    world, onto = _load_world()
    expr = parse_class_expression(query, onto)
    with onto:
        if subject:
            entity, kind = resolve_subject(subject, onto)
        else:
            entity, kind = Thing, 'class'
        if kind == 'individual':
            target = entity
        else:
            # fresh individual that is a member of the subject class
            target = Thing(f'query_subject_{uuid.uuid4().hex[:8]}', namespace=onto)
            if entity is not Thing:
                target.is_a.append(entity)
        target.is_a.append(Not(expr) if negate else expr)
    return _is_consistent(world)


def evaluate_query(query: str, subject: str = '') -> str:
    """Return ENTAILED, CONTRADICTED or UNKNOWN; raises ManchesterError on invalid queries."""
    _check_base_ontology()
    query, subject = query.strip(), (subject or '').strip()
    # parse once up front so syntax errors are reported before the reasoner runs
    _, onto = _load_world()
    parse_class_expression(query, onto)
    if subject:
        resolve_subject(subject, onto)

    if not _consistent_with(query, subject, negate=False):
        return CONTRADICTED          # O |= not Q(s)
    if not _consistent_with(query, subject, negate=True):
        return ENTAILED              # O |= Q(s)
    return UNKNOWN                   # open world: neither follows


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
    # Environment problems (no Java, inconsistent base ontology) must stop the run instead of
    # silently turning every claim into an error or an Unknown.
    _check_base_ontology()
    try:
        outcome = evaluate_query(query, subject)
    except ManchesterError as e:
        result = f'{ERROR}: {e}'
    except Exception as e:  # never crash the agent; report the problem so it can retry
        result = f'{ERROR}: {type(e).__name__}: {e}'
    else:
        result = outcome
    print(f'  -> {result}', flush=True)
    return result
