from configparser import ConfigParser
from pathlib import Path
from owlready2 import World

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
ONTOLOGY_PATH = config['ONTOLOGY']['PATH']

def read_ontology():
    world = World()
    ontology = world.get_ontology(BASE_DIR / ONTOLOGY_PATH).load()
    classes = list(world.classes())
    properties = list(world.properties())
    data_properties = list(world.data_properties())
    return f'The ontology has the following classes:\n{classes}\nThe following properties:\n{properties}\nAnd the following data properties:\n{data_properties}'
VERIFICATION_PROMPT += f'\n{read_ontology}'

TRANSLATION_PROMPT = config['PROMPTS']['TRANSLATION']

LOG_PATH = BASE_DIR / config['LOGGING']['PATH']

TWEETS_PATH = BASE_DIR / config['TWEETS']['PATH']

USER_PROMPT = BASE_DIR / config['PROMPTS']['USER']

BELIEFS = {'McDonalds': "Americans basically live on McDonald's, and everyone knows it's all unhealthy junk.",
'Christmas': "Christmas is the holiday everyone celebrates at the end of the year, and it's all about presents.",
'Netflix': "Everything worth watching on Netflix is a Netflix original, and nobody watches anything else."}