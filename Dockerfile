FROM python:3.10

RUN pip install --upgrade pip

WORKDIR tasks
COPY requirements.txt /tasks/requirements.txt

RUN pip install -r /tasks/requirements.txt --default-timeout=100  # timeout対策