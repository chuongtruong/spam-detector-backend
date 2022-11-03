install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	docker run --rm -i hadolint/hadolint < Dockerfile
	pylint --disable=R,C,W1203,W0702 app.py

test:
	python -m pytest -vv --cov=app test_app.py

build:
	docker build -t quick-read-mvp:latest .

run:
	docker run -p 8080:8080 quick-read-mvp

all: install lint test

## multi-platform setup for docker build: https://medium.com/geekculture/docker-build-with-mac-m1-d668c802ab96
## build and push to docker with buildx: docker buildx build --push --tag sophietruong92/quick-read-mvp:latest --platform=linux/arm64,linux/amd64 .

