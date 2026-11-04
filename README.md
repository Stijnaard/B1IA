# B1IA – ontology-checked tweet summarisation

Run `python Agent/Agent.py` (from the repo root or `Agent/`). It runs the agent for every user prompt in
`user_prompts/` on every topic in `inputs/clean/`, then the baseline (SummaryAgent without a user prompt).
Logs go to `Agent/logs/`. HermiT needs Java: either `java` on the PATH or `[REASONER] JAVA` in `config.ini`.

## Changes on branch `stijn`

In the earlier runs every query came back `not_verifyable`. As a result the revision loop never ran, and the
only difference from the baseline was the user prompt. There were two causes:

1. **The query agent never saw the ontology.** `configs.py` appended the *function object* `read_ontology`
   (`<function read_ontology at 0x…>`) to the Agent 2 prompt instead of calling it. The QueryAgent therefore
   invented terms (`McDonald's`, `NetflixSubscription`, `ChristmasBreak`, …) that do not exist in the ontology.
2. **Every failed check counted as "consistent".** `check_query` caught any SPARQL error, printed it, and then
   ran the reasoner on the *unchanged* ontology, which is consistent. So the original query and all negated
   queries "passed", which always gave `not_verifyable`. The SPARQL almost always failed: `graph.update()`
   only accepts INSERT/DELETE, the LLM mostly produced SELECT/ASK queries (often inside code fences), and the
   253 KB RDF (~63k tokens) did not fit in the 16k context of the translating LLM.

### What changed

| File | Change |
|---|---|
| `Agent/configs.py` | `read_ontology()` is now called and lists the classes, object/data properties (with domain → range) and individuals (~3k tokens). Adds `OLLAMA_NUM_CTX` and `JAVA_EXE`. |
| `Agent/manchester.py` | **New.** Parser for Manchester OWL class expressions into owlready2 constructs. Unknown terms return an error with close matches, so the QueryAgent can retry. |
| `Agent/tool.py` | Rewritten Ontology Tool (Eq. 3): asserts `Q(s)` and `¬Q(s)` in two isolated copies of the ontology and lets HermiT check consistency → `Entailed` / `Contradicted` / `Unknown`, or `Error: …`. Takes a `subject` individual (empty = satisfiability check, as in Scenarios 2 and 5). |
| `Agent/Agent.py` | Ollama `num_ctx` set (the default silently truncated the prompt). The state has a new `unknown` list; `not_verifiable` now holds the claims the QueryAgent itself labels Not-Verifiable. Violations carry the sentence number and claim text. The revision prompt now passes the *previous* summary `S_i` (it passed the one before that, which was empty). The system prompt is no longer sent twice to the QueryAgent. A recursion guard prevents a crash when the QueryAgent keeps retrying. `pick_best_summary` keeps all iterations in the log. |
| `Agent/System Prompt - Agent 1.md` | Revision section updated to the report's Appendix B.1. |
| `Agent/System Prompt - Agent 2.md` | Updated to Appendix B.2, plus a section on how to call the tool (subject, sentence, claim, Manchester examples from Table 2) and a fixed output format for Not-Verifiable claims. |

### Deviation from the report: TableauAgent and SparQLAgent replaced by code

The report (Fig. 1, §2.3) has two LLM nodes inside the Ontology Tool. The TableauAgent writes the NNF of the
query and the SparQLAgent turns it into SPARQL. Both are replaced by `manchester.py` plus a direct owlready2
assertion. The checking itself is still done by owlready2 + HermiT exactly as in Eq. 3. Reasons:

- **HermiT already is a tableau reasoner.** It computes the negation normal form and the branching itself,
  so an LLM-made NNF adds nothing, and any error in it (for example a wrong negation) silently makes the
  verdict wrong. Asserting `not Q` and letting HermiT expand it is correct by construction.
- **The SPARQL step could not work.** Asserting a class expression in RDF needs `owl:Restriction` /
  `owl:complementOf` blank-node structures. A 9B model cannot write these reliably, and it never saw the full
  ontology. owlready2 produces exactly those triples from the parsed expression.
- **Determinism and cost.** §2.4 argues for a formal reasoner because it is deterministic and explainable.
  With LLMs in the loop the verdict was neither. Each check now takes about 0.5–5 s and needs no LLM calls,
  where it used to take two LLM calls with a 63k-token prompt.

The five scenarios of Table 2 all give the outcome stated in the report (checked with HermiT):

| Subject | Query | Outcome |
|---|---|---|
| `company:mcdonalds` | `activeIn value country:netherlands` | Entailed |
| `company:mcdonalds` | `headquarteredIn value country:uk` | Contradicted |
| — | `CoeliacPerson and canEat value food:burger` | Contradicted |
| `food:salad` | `FastFood` | Contradicted |
| `holiday:ethiopianChristmas` / `holiday:christmas` | `celebratedIn value country:ethiopia` | Entailed / Unknown |
| — | `Person and hasAge value 16 and isMarriedTo some Person` | Contradicted |

`Agent/System Prompt - Agent 3.md` and `[PROMPTS] TRANSLATION` are no longer used.
