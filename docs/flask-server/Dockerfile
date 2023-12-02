FROM python:3.10.11

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

RUN apt-get update

RUN apt-get install libgl1 -y

COPY . /app

EXPOSE 5000

CMD ["python", "app.py"]
