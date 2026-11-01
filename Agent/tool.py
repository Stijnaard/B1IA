from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage
from owlready2 import World, OwlReadyInconsistentOntologyError, sync_reasoner

from configs import OLLAMA_MODEL, ONTOLOGY_PATH, BASE_DIR

@tool
def check_ontology(query: str) -> str:
    "checks if the query holds in the ontology, returns verified when entailed, violation when a counterexample or contradiction is found and not_verifyable when it holds only due to the open world assumption" 
    print(f'checking query: {query}')
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        reasoning=True,
        num_ctx=8192*2,
        client_kwargs={'timeout': 900}
    )

    # Get negated normal form
    system_prompt= "The user prompt will be a manchester OWL query, your job is to respond with the negation normal form. Use the Tableau algorithm. In your output only respond with the negated query in manchester owl syntax, if you need to respond with multiple queries due to the tablue algorithm seperate them with comma's."
    response = llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(query)
    ])
    queries = [query]+ response.content.split(',')
    print(f'nnf queries: {queries}')
    # Translate into sparql
    sparQL_queries = []
    for q in queries:
        system_prompt2 = "The user prompt will be a manchester OWL query, it is your job to translate it into SparQL corresponding with the rdf file below. In your output **only** respond with the sparQL query.\n RDF:\n{rdf}"
        response2 = llm.invoke([
            SystemMessage(system_prompt2),
            HumanMessage(q)
        ])
        sparQL_queries.append(response2.content)
    print(f'sparQL queries: {sparQL_queries}')
    #Check queries

    if not check_query(sparQL_queries[0]):
        return 'violation'
    for sparQL_query in sparQL_queries[1:]:
        if not check_query(sparQL_query):
            return 'verified'
    return 'not_verifyable' 
    

def check_query(query):
    print(f'feeding query to ontology: {query}')
    #Returns False on contradiction, True otherwise
    isolated_world = World()
    onto = isolated_world.get_ontology(str(BASE_DIR/ONTOLOGY_PATH)).load()

    try:
        sync_reasoner(isolated_world)
    except OwlReadyInconsistentOntologyError as e:
        raise Exception('the base ontology is inconsitent') from e

    graph = isolated_world.as_rdflib_graph()
    try:
        with onto:
            graph.update(query)
    except Exception as e:
        print(f'invalid sparql: {query}')

    try:
        sync_reasoner(isolated_world)
    except OwlReadyInconsistentOntologyError as e:
        return False
    
    return True

