FROM python:3.9

COPY ./requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY . /app

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
