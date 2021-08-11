# Multilevel Diacritizer
### *Simple extensible deep learning model for automatic Arabic diacritization*

[![Continious Integration](https://github.com/Hamza5/multilevel-diacritizer/actions/workflows/CI.yml/badge.svg)](https://github.com/Hamza5/multilevel-diacritizer/actions/workflows/CI.yml)

## About

This repository contains the code of the tool developed for the research titled
[*Simple extensible deep learning model for automatic Arabic diacritization*](#)
[*Publication pending*]. This is a command line tool that allow training and loading
a parametrized deep learning model. It supports restoring the diacritics of
a text fed through the command line. Also, it allows launching a web server that
accept plain text requests through the POST method and returns the diacritized text.
This web server displays the interface of the web application when receiving a GET request.

## Download

You can get the code contained in this repository using the green `Code` GitHub button
above or by the standard `git clone` command.

## Running

### Option 1: Inside a Docker container

If you prefer using [Docker](https://www.docker.com/), you can easily build an image of
this repository using the included `Dockerfile`.

```bash
$ cd multilevel-diacritizer/
$ docker build -t multilevel-diacritizer:latest ./
$ docker run --rm -d --name multilevel-diacritizer -p 8000:8000 multilevel-diacritizer
```

You can then access to the web application interface through `http://localhost:8000`.

The application  is located inside the `/app` directory which is the default working
directory. You can run the command line application from there.

```bash
$ docker exec -it multilevel-diacritizer /bin/bash
root@multilevel-diacritizer:/app# export PYTHONPATH=.
root@multilevel-diacritizer:/app# source .venv/bin/activate
(.venv) root@multilevel-diacritizer:/app# python multilevel_diacritizer/multi_level_diacritizer.py --help
```

### Option 2: Directly on the host

To use this application directly without Docker, you need first to install [Poetry](https://python-poetry.org/),
a Python package manager that we use in order to install the dependencies of this project.

```bash
$ pip install poetry
```

Then you have to install the dependencies of the project using Poetry. Poetry will automatically
create a virtual environment to install the dependencies in. Once installed, you can spawn a
Poetry shell inside the configured virtual environment and run the command line application
from there.

```bash
$ cd multilevel-diacritizer/
$ poetry install --no-root --no-dev
$ poetry shell
$ python multilevel_diacritizer/multi_level_diacritizer.py --help
```

In this method, the web application is not started automatically. You can launch it using the
`server` command, then you can access it on `http://localhost:8000`:

```bash
$ python multilevel_diacritizer/multi_level_diacritizer.py server
```

## Deployment

This repository contains a small web application optimized for deployment on Heroku.
It gives the user a simple web interface to input any text, then send it to the backend
to be diacritized by our deep learning model, then the result will be displayed
on the web interface. You can deploy it to your Heroku account by clicking the
following button (you can make an account if you do not have, it is free).

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)