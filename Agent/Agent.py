from __future__ import annotations #allows for recursive type annotations

import json
from configparser import ConfigParser
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from time import time
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
#from tool import check_ontology


# Load in configs
config = ConfigParser()

BASE_DIR = Path(__file__).resolve().parent.parent # root dir of repo
config.read(BASE_DIR / "config.ini")
GROQ_API_KEY = config['GROQ']['GROQ_API_KEY']
CROQ_MODEL = config['GROQ']['MODEL']

OLLAMA_MODEL = config['OLLAMA']['MODEL']

sum_prompt_path = BASE_DIR / config['PROMPTS']['SUMMARIZATION']
with open(sum_prompt_path, 'r', encoding='utf-8') as f:
    SUMMARIZATION_PROMPT = f.read()
ver_prompt_path = BASE_DIR / config['PROMPTS']['VERIFICATION']
with open(ver_prompt_path, 'r', encoding='utf-8') as f:
    VERIFICATION_PROMPT = f.read()
LOG_PATH = BASE_DIR / config['LOGGING']['PATH']
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
            client_kwargs={'timeout': 900}
        )

    else:
        raise NotImplementedError(message = f'llm_type {llm_type} has not been implemented yet')
    return llm

# setup dataclasses

class Verification(BaseModel):
    verified: list[str] = Field(default_factory=list, description="Claims verified to hold in the ontology")
    violations: list[str] = Field(default_factory=list, description="Claims where a contradiction or counterexample exists")
    not_verifiable: list[str] = Field(default_factory=list, description="Claims that hold only due to the open world assumption")

@dataclass
class State:
    user_prompt: str
    tweets: Path
    summary_count: int = 0
    summary: str = ""
    queries: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    not_verifiable: list[str] = field(default_factory=list)
    history: State|None = None
    # ^turns state into a linked list, bad for performance but will be fine for small sizes like here


# tools
import random
@tool
def check_ontology(query: str) -> str:
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
        HumanMessage(build_prompt(state.user_prompt, state.tweets))
    ])

    # make new state, storing historical state, incrementing the sumarycount and adding the LLM response
    return State(state.user_prompt, state.tweets, state.summary_count+1, response.content, history = state)


def build_prompt(user_prompt: str, tweets: Path):
    """"Assumes tweets is a complete filepath"""
    with open(tweets, 'r', encoding='utf-8') as f:
        return f"{user_prompt}\n{f.readlines()}"

def extract_queries(messages) -> list[dict]:
    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    return [
        {"query": tc["args"], "result": results.get(tc["id"])}
        for m in messages if isinstance(m, AIMessage)
        for tc in m.tool_calls
    ]

def classify_queries(queries: list[dict]) -> Verification:
    res = Verification()
    for q in queries:
        query = str(q['query'].get('query', q['query']))
        if q['result'] == 'verified':
            res.verified.append(query)
        elif q['result'] == 'violation':
            res.violations.append(query)
        elif q['result'] == 'not_verifyable':
            res.not_verifiable.append(query)
    return res

def verify(state: State, llm):
    checker_agent = create_agent(
    llm,
    tools=[check_ontology],
    system_prompt=VERIFICATION_PROMPT
    )
    raw_result = checker_agent.invoke({'messages': [
            SystemMessage(VERIFICATION_PROMPT),
            HumanMessage(state.summary)
        ]})

    result = classify_queries(extract_queries(raw_result['messages']))
    return {
        'verified': result.verified,
        'violations': result.violations,
        'not_verifiable': result.not_verifiable,
        'queries': extract_queries(raw_result['messages']),
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
    return helper(state, state)

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
    for mode, chunk in app.stream(init_state, steam_mode=['updates', 'values']):
        if mode == 'updates':
            for node in chunk:
                print(f"[{time.time() - t0:6.0f}s] finished node: {node}", flush=True)
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

def main():
    llm = load_llm('ollama')
    app = build_graph(llm)

    # print graph
    png_bytes = app.get_graph(xray=True).draw_mermaid_png()
    with open('graph.png', 'wb') as f:
        f.write(png_bytes)
    
    t0 = time()
    ran_state = run_agent(app, 'TODO', Path(BASE_DIR / 'inputs/Netflix_clean.txt'))
    t = time()-t0
    print(f'ran in {t}')

    print_summaries(ran_state)
    
    write_run_log(LOG_PATH, ran_state, duration = t)

if __name__ == "__main__":
    main()


