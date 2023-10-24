FROM svizor/zoomcamp-model:3.10.12-slim

ENV PYTHONUNBUFFERED=TRUE

RUN pip --no-cache-dir install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy --system && \
    rm -rf /root/.cache

COPY ["3docker.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "3docker:app"]
#ENTRYPOINT ["waitress-serve", "--host=0.0.0.0", "--port=9696", "churn_serving:app"]

