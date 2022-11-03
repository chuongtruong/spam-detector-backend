FROM python:3.8.8-slim-buster
# FROM arm64v8/python:latest

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY . app.py /app/

# Install packages from requirements.txt
# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
# ENTRYPOINT [ "flask"]
# CMD [ "run", "--host" ,"0.0.0.0", "--port", "8080" ]

