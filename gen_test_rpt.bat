coverage run --source=mathtools --parallel-mode -m pytest
coverage combine
coverage xml -i
coverage report -m
