# syntax=docker/dockerfile:1

FROM python:3.7.10-buster

ENV APPDIR=app/
WORKDIR $APPDIR
ENV PYTHONPATH=$APPDIR
ENV POETRY_VIRTUALENVS_CREATE=true
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH=$PATH:.venv/bin/

RUN pip install poetry

COPY pyproject.toml ./
COPY poetry.lock ./
RUN  poetry install --no-root --no-dev --no-interaction
COPY multilevel_diacritizer/ ./multilevel_diacritizer
COPY multilevel_diacritizer_ui/build/web/ ./multilevel_diacritizer_ui/build/web
COPY params/ ./params

EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "multilevel_diacritizer.multi_level_diacritizer:create_server_app()"]
