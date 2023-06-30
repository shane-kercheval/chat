####
# DOCKER
####
docker_build:
	docker compose -f docker-compose.yml build

docker_run: docker_build
	docker compose -f docker-compose.yml up

docker_down:
	docker compose down --remove-orphans

docker_rebuild:
	docker compose -f docker-compose.yml build --no-cache

docker_bash:
	docker compose -f docker-compose.yml up --build bash

####
# Project
####
linting:
	ruff check source

tests: linting

streamlit:
	python -m streamlit run source/app.py
