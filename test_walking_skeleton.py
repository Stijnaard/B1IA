from CI_project import LLM

import pytest

pytest.mark.unit
def test_walking_skeleton():
    model: LLM = LLM("liquid/lfm2.5-1.2b", "http://localhost:1234/v1")