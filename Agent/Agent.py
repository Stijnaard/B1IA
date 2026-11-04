from __future__ import annotations #allows for recursive type annotations

import json
import traceback
from configparser import ConfigParser
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from time import time
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END
from owlready2 import World, sync_reasoner, OwlReadyInconsistentOntologyError
from pydantic import BaseModel, Field

from configs import BASE_DIR, GROQ_API_KEY, CROQ_MODEL, OLLAMA_MODEL, OLLAMA_NUM_CTX, SUMMARIZATION_PROMPT, VERIFICATION_PROMPT, ONTOLOGY_PATH, LOG_PATH, TWEETS_PATH, USER_PROMPT, BELIEFS
from tool import check_ontology, ENTAILED, CONTRADICTED, UNKNOWN, ERROR


# setup llm connection

def load_llm(llm_type: str = 'groq'):
    if llm_type == 'groq':
        llm = ChatGroq(
        model = CROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0
        )
    elif(llm_type=='ollama'):
        llm = ChatOllama(
            model = OLLAMA_MODEL,
            base_url='http://127.0.0.1:11434', #default, here to change easily if needed
            temperature=0,
            reasoning=False, #can be turned on but will be slower
            num_ctx=OLLAMA_NUM_CTX, # Ollama's default context window silently truncates the prompt
            client_kwargs={'timeout': 900}
        )

    else:
        raise NotImplementedError(message = f'llm_type {llm_type} has not been implemented yet')
    return llm

# setup dataclasses

class Verification(BaseModel):
    verified: list[str] = Field(default_factory=list, description="Claims entailed by the ontology")
    violations: list[str] = Field(default_factory=list, description="Claims contradicted by the ontology")
    unknown: list[str] = Field(default_factory=list, description="Claims neither entailed nor contradicted (open world assumption)")
    not_verifiable: list[str] = Field(default_factory=list, description="Claims QueryAgent could not express with the ontology vocabulary")

class SparQLQueries(BaseModel):
    negated: list[str] = Field(default_factory=list, description='List of negated sparql queries')
    regular: list[str] = Field(default_factory=list, description='List of regular sparql qieries')


@dataclass
class State:
    user_prompt: str
    tweets: Path
    summary_count: int = 0
    summary: str = ""
    queries: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    not_verifiable: list[str] = field(default_factory=list)
    history: State|None = None
    # ^turns state into a linked list, bad for performance but will be fine for small sizes like here


# tools
import random
@tool
def check_ontology_dumm(query: str) -> str:
    "checks if the query holds in the ontology, returns verified when entailed, violation when a counterexample or contradiction is found and not_verifyable when it holds only due to the open world assumption" 
    x = random.randint(0,2)
    if x == 0:
        return "verified"
    elif x ==1:
        return 'violation'
    else:
        return 'not_verifyable'

    # need to catch actual error and return them as text for the agent to retry failed queries


# work nodes

def summarize(state: State, llm):
    # send prompt to LLM (no agent needed, this is a single linear call)
    response = llm.invoke([
        SystemMessage(SUMMARIZATION_PROMPT),
        HumanMessage(build_prompt(state))
    ])

    # make new state, storing historical state, incrementing the sumarycount and adding the LLM response
    return State(state.user_prompt, state.tweets, state.summary_count+1, response.content, history = state)


def build_prompt(state: State):
    """"Assumes tweets is a complete filepath"""
    with open(state.tweets, 'r', encoding='utf-8') as f:
        tweets = f.readlines()

    res = f"{state.user_prompt}\n{tweets}"
    if len(state.violations) != 0 and state.summary_count != 0:
        # state is the verified state of the previous iteration: its summary is S_i and its
        # violations are C_i (Eq. 4 in the report)
        res += f"\n\nYour previous summary was:\n{state.summary}"
        res += "\n\nThe ontology reasoner found the following claims in the previous summary to be contradicted:"
        for violation in state.violations:
            res += f"\n- {violation}"
    return res

def extract_queries(messages) -> list[dict]:
    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    return [
        {"query": tc["args"], "result": results.get(tc["id"])}
        for m in messages if isinstance(m, AIMessage)
        for tc in m.tool_calls
    ]

def describe_claim(args: dict) -> str:
    """Readable description of a checked claim, used in the state and in the revision prompt."""
    query = args.get('query', '')
    subject = args.get('subject') or ''
    check = f'{subject} is an instance of ({query})' if subject else f'something can be ({query})'
    text = f'"{args["claim"]}" ' if args.get('claim') else ''
    sentence = f'Sentence {args["sentence"]}: ' if args.get('sentence') else ''
    return f'{sentence}{text}[checked: {check}]'

def classify_queries(queries: list[dict]) -> Verification:
    res = Verification()
    for q in queries:
        args = q['query'] if isinstance(q['query'], dict) else {'query': str(q['query'])}
        result = str(q['result'] or '')
        if result.startswith(ENTAILED):
            res.verified.append(describe_claim(args))
        elif result.startswith(CONTRADICTED):
            res.violations.append(describe_claim(args))
        elif result.startswith(UNKNOWN):
            res.unknown.append(describe_claim(args))
        # results starting with ERROR are retried by the QueryAgent and stay visible in `queries`
    return res

def extract_not_verifiable(messages) -> list[str]:
    """Not-Verifiable claims listed by the QueryAgent in its final answer."""
    final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    if final is None or not isinstance(final.content, str):
        return []
    claims = []
    for line in final.content.splitlines():
        line = line.replace('**', '').replace('`', '').strip().lstrip('-* ')
        if line.upper().startswith('NOT-VERIFIABLE:'):
            claims.append(line.split(':', 1)[1].strip())
    return claims

def verify(state: State, llm):
    checker_agent = create_agent(
    llm,
    tools=[check_ontology],
    system_prompt=VERIFICATION_PROMPT
    )
    # the system prompt is already set by create_agent; QueryAgent only sees S_i and O
    messages = [HumanMessage(state.summary)]
    try:
        # stream so the claims checked so far are kept if the agent keeps retrying failing queries
        for chunk in checker_agent.stream({'messages': messages}, {'recursion_limit': 40}, stream_mode='values'):
            messages = chunk['messages']
    except GraphRecursionError:
        print('QueryAgent hit the recursion limit; using the queries checked so far', flush=True)
    raw_result = {'messages': messages}

    queries = extract_queries(raw_result['messages'])
    result = classify_queries(queries)
    return {
        'verified': result.verified,
        'violations': result.violations,
        'unknown': result.unknown,
        'not_verifiable': extract_not_verifiable(raw_result['messages']),
        'queries': queries,
    }

def pick_best_summary(state: State):
    """picks the best summary based on number of violations"""
    def helper(current_state: State, best_state: State) -> State:
        if current_state.history is None:
            return best_state
        if len(current_state.violations) <= len(best_state.violations):
            return helper(current_state.history, current_state)
        else:
            return helper(current_state.history, best_state)
    best = helper(state, state)
    # keep every iteration in the history (the log would otherwise lose the later summaries)
    return replace(best, history=state)

def route(state: State, max_tries: int = 3): 
    "both routing diamonds in fig 1"
    if len(state.violations) == 0:
        return END
    elif state.summary_count < max_tries:
        return 'summarize'
    else:
        return 'pick_best_summary'

# langchain graph
def build_graph(llm):
    graph = StateGraph(State)
    graph.add_node('summarize', partial(summarize, llm = llm))
    graph.add_node('verify_summary', partial(verify, llm = llm))

    graph.add_node('pick_best_summary', pick_best_summary)
    graph.add_edge(START, 'summarize')
    graph.add_edge('summarize', 'verify_summary')
    graph.add_conditional_edges('verify_summary', route)
    graph.add_edge('pick_best_summary', END)

    app = graph.compile()
    return app


def run_agent(app, user_prompt: str, tweets: Path) -> State:
    init_state = State(user_prompt,tweets)
    out_state = State(**app.invoke(init_state))
    return out_state

def run_agent_with_progress(app, user_prompt: str, tweets: Path) -> State:
    init_state = State(user_prompt,tweets)

    final = None
    t0 = time()
    for mode, chunk in app.stream(init_state, stream_mode=['updates', 'values']):
        if mode == 'updates':
            for node in chunk:
                print(f"[{time() - t0:6.0f}s] finished node: {node}", flush=True)
        else:
            final = chunk
    return State(**final)

def print_summaries(state: State):
    print(f'Summary {state.summary_count}: \n{state.summary}')
    if state.history is None:
        return
    else:
        return print_summaries(state.history)

def write_run_log(log_path: Path, state: State, name: str|None = None, **meta):
    record = {**meta, 'state': asdict(state)}
    if name is not None:
        file_path = log_path/name
    else:
        file_path = log_path/Path(f'{datetime.now().isoformat(timespec='seconds')}.json')
    file_path.write_text(
        json.dumps(record, default=str, indent=2, ensure_ascii=False),
        encoding = 'utf-8'
    )

# def main():
#     llm = load_llm('ollama')
#     app = build_graph(llm)

#     # print graph
#     png_bytes = app.get_graph(xray=True).draw_mermaid_png()
#     with open('graph.png', 'wb') as f:
#         f.write(png_bytes)
    
#     t0 = time()
#     ran_state = run_agent(app, 'TODO', Path(BASE_DIR / 'inputs/Netflix_clean.txt'))
#     t = time()-t0
#     print(f'ran in {t}')

#     print_summaries(ran_state)
    
#     write_run_log(LOG_PATH, ran_state, duration = t)


def fill_user_prompt(user_prompt: str, topic: str, k: int = 5) -> str:
    return user_prompt.replace('{topic}', topic).replace('{belief}', BELIEFS[topic]).replace('{k}', str(k))

def sumarize_file(file: Path, user_prompt: str, upc: int, k: int = 5, llm: str = 'ollama'):
    llm =  load_llm(llm)
    app = build_graph(llm)
    user_prompt = fill_user_prompt(user_prompt, file.stem, k)
    print(f'running with file: {file} and prompt: {user_prompt}')
    t0 = time()
    ran_state = run_agent(app, user_prompt, file)
    t = time()-t0
    print(f'ran in {t}')
    print_summaries(ran_state)
    write_run_log(LOG_PATH, ran_state, name= f'{file.stem}_{upc}.json', duration = t)

def sumarize_folder(folder: Path, user_prompt:str, upc: str, k: int = 5, llm: str = 'ollama'):
    for file in folder.iterdir():
        if file.is_file():
            sumarize_file(file, user_prompt, upc, k, llm)

def use_all_prompts(prompts_folder: Path, tweets_folder: Path, k: int = 5, llm: str = 'ollama'):
    for file in prompts_folder.iterdir():
        if file.is_file():
            with open(file, 'r') as f:
                user_prompt = f.read()
            sumarize_folder(tweets_folder, user_prompt, file.stem, k, llm)

def main():
    use_all_prompts(USER_PROMPT, TWEETS_PATH, 5)

def baseline():
    llm = load_llm('ollama')
    for file in TWEETS_PATH.iterdir():
        if file.is_file():
            state = State("", file)
            t0 = time()
            summary = summarize(state, llm)
            t = time()-t0
            write_run_log(LOG_PATH, summary, f'{file.stem}_baseline.json', duration = t)

if __name__ == "__main__":
    main()
    baseline()
