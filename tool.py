# Run this file by using the cmd: 'python tool.py IAOntology.rdf negated_claim.sparql'
# where the IAOntology.rdf should correspond to the ontology file, and negated_claim.sparql should correspond to a file containing a valid sparql version of the negated claim you're trying to confirm





import argparse
from owlready2 import get_ontology, World, sync_reasoner, OwlReadyInconsistentOntologyError


# assumes a sparql string for negated claim

def check_claim(onto_path: str, negated_claim: str) -> (None|bool):
    #specify an isloated world for the modification to the ontology to be setup in
    isolated_world = World()
    onto = isolated_world.get_ontology(onto_path).load()
    
    
    try:
        sync_reasoner(isolated_world)
    except OwlReadyInconsistentOntologyError as e:
        raise Exception('The base ontology is inconcistent') from e

    graph = isolated_world.as_rdflib_graph()
    with onto:
        graph.update(negated_claim)

    try:
        sync_reasoner(isolated_world)
    except OwlReadyInconsistentOntologyError as e:
        return True # as the negation is inconsitent, the claim must be consistent

    return None # the claim may be true, may not be
    



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("ontology")
    parser.add_argument("claim", help="Path to SPARQL file")

    args = parser.parse_args()

    with open(args.claim, "r") as f:
        negated_claim = f.read()

    result = check_claim(args.ontology, negated_claim)

    print(result)


if __name__ == "__main__":
    main()


