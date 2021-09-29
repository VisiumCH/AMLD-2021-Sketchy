FROM python:3.7

WORKDIR /app

# Create and activate the virtual environment
ENV VIRTUAL_ENV=/app/env
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY ./requirements.txt requirements.txt
COPY setup.py setup.py

RUN apt-get update && apt-get install -y libgl1-mesa-dev

RUN pip install --upgrade pip; \
    pip install setuptools wheel \
    pip install -r requirements.txt

COPY ./src src

RUN pip install -e .

CMD python src/api/server.py