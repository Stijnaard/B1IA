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


def read_ontology():
    """Ontology structure O (classes, properties and individuals) given to the QueryAgent,
    plus the sets of names that the ontology tool accepts."""
    world = World()
    world.get_ontology(str(BASE_DIR / ONTOLOGY_PATH)).load()
    classes = sorted(c.name for c in world.classes())
    object_properties = list(world.object_properties())
    data_properties = list(world.data_properties())
    return f'The ontology has the following classes:\n{classes}\nThe following properties:\n{properties}\nAnd the following data properties:\n{data_properties}'
VERIFICATION_PROMPT += f'\n{read_ontology()}'

TRANSLATION_PROMPT = config['PROMPTS']['TRANSLATION']

LOG_PATH = BASE_DIR / config['LOGGING']['PATH']

TWEETS_PATH = BASE_DIR / config['TWEETS']['PATH']

USER_PROMPT = BASE_DIR / config['PROMPTS']['USER']

BELIEFS = {'McDonalds': "Americans basically live on McDonald's, and everyone knows it's all unhealthy junk.",
'Christmas': "Christmas is the holiday everyone celebrates at the end of the year, and it's all about presents.",
'Netflix': "Everything worth watching on Netflix is a Netflix original, and nobody watches anything else."}