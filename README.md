Run `python Agent/Agent.py` (from the repo root or `Agent/`). It runs the agent for every user prompt in
`user_prompts/` on every topic in `inputs/clean/`, then the baseline (SummaryAgent without a user prompt).
Logs go to `Agent/logs/`. 
HermiT runs through [owlapy](https://github.com/dice-group/owlapy) and needs Java:
either `JAVA_HOME` set, or `[REASONER] JAVA` in `config.ini`.
Results are stored in the log folder (change in the config.ini)
Testsed on Python 3.14.8


Run `python run_all.py` (from the Agent_evaluation) to run the evaluation.
Needs to run on python 3.11
Reads files in the outputs folder
