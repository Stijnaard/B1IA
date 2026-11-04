from configparser import ConfigParser
from pathlib import Path
from owlready2 import World

config = ConfigParser()

BASE_DIR = Path(__file__).resolve().parent.parent # root dir of repo
config.read(BASE_DIR / "config.ini", encoding="utf-8")
GROQ_API_KEY = config['GROQ']['GROQ_API_KEY']
CROQ_MODEL = config['GROQ']['MODEL']

OLLAMA_MODEL = config['OLLAMA']['MODEL']

sum_prompt_path = BASE_DIR / config['PROMPTS']['SUMMARIZATION']
with open(sum_prompt_path, 'r', encoding='utf-8') as f:
    SUMMARIZATION_PROMPT = f.read()
ver_prompt_path = BASE_DIR / config['PROMPTS']['VERIFICATION']
with open(ver_prompt_path, 'r', encoding='utf-8') as f:
    VERIFICATION_PROMPT = f.read()
ONTOLOGY_PATH = config['ONTOLOGY']['PATH']

def _entity_name(e) -> str:
    return getattr(e, 'name', None) or getattr(e, '__name__', None) or str(e)


def _signature(prop) -> str:
    """'name (Domain -> Range)' so the QueryAgent knows how a property can be used."""
    domain = ', '.join(_entity_name(d) for d in prop.domain)
    rng = ', '.join(_entity_name(r) for r in prop.range)
    if domain or rng:
        return f'{prop.name} ({domain or "Thing"} -> {rng or "Thing"})'
    return prop.name


def read_ontology() -> str:
    """Ontology structure O (classes, properties and individuals) given to the QueryAgent."""
    world = World()
    world.get_ontology(str(BASE_DIR / ONTOLOGY_PATH)).load()
    classes = sorted(c.name for c in world.classes())
    object_properties = sorted(_signature(p) for p in world.object_properties())
    data_properties = sorted(_signature(p) for p in world.data_properties())
    individuals = sorted(i.name for i in world.individuals())
    return ('## Ontology structure\n\n'
            f'Classes: {", ".join(classes)}\n\n'
            f'Object properties (domain -> range): {"; ".join(object_properties)}\n\n'
            f'Data properties (domain -> range): {"; ".join(data_properties)}\n\n'
            f'Individuals: {", ".join(individuals)}\n')


ONTOLOGY_STRUCTURE = read_ontology()
VERIFICATION_PROMPT += f'\n\n{ONTOLOGY_STRUCTURE}'

# Optional path to a java executable for the HermiT reasoner (e.g. the JRE bundled with Protege).
# Leave empty to use `java` from the PATH.
JAVA_EXE = config.get('REASONER', 'JAVA', fallback='').strip()

# Context window for Ollama; its default (2048-4096 tokens) silently truncates the tweets and
# the ontology structure.
OLLAMA_NUM_CTX = config.getint('OLLAMA', 'NUM_CTX', fallback=16384)

LOG_PATH = BASE_DIR / config['LOGGING']['PATH']

TWEETS_PATH = BASE_DIR / config['TWEETS']['PATH']

USER_PROMPT = BASE_DIR / config['PROMPTS']['USER']

BELIEFS = {'McDonalds': "Americans basically live on McDonald's, and everyone knows it's all unhealthy junk.",
'Christmas': "Christmas is the holiday everyone celebrates at the end of the year, and it's all about presents.",
'Netflix': "Everything worth watching on Netflix is a Netflix original, and nobody watches anything else."}