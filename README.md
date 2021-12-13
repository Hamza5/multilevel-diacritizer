# Multilevel Diacritizer
### *Simple extensible deep learning model for automatic Arabic diacritization*

[![Continious Integration](https://github.com/Hamza5/multilevel-diacritizer/actions/workflows/CI.yml/badge.svg)](https://github.com/Hamza5/multilevel-diacritizer/actions/workflows/CI.yml)

## About

This repository contains the code of the tool developed for the research titled
[*Simple extensible deep learning model for automatic Arabic diacritization*](https://dl.acm.org/doi/10.1145/3480938?cid=99659913905).
This is a command line tool that allow training and loading
a parametrized deep learning model. It supports restoring the diacritics of
a text fed through the command line.

It includes a simple web application that gives the user a simple web interface
to input any text, then send it to the backend  to be diacritized by our deep
learning model, then the result will be displayed on the web interface.

Also, it allows launching a web server that accept plain text requests through the POST method and returns the
diacritized text. This web server displays the interface of the web application when receiving a GET request.

## Download

You can get the code contained in this repository using the ![Code](https://img.shields.io/badge/-Code-forestgreen)
button above or by the standard `git clone` command.

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

## Functionalities

The command line application provides three main commands: `train`, `diacritization`, and `server`. There is a less
important command `confusion-matrix` that is used to generate the confusion matrix from a predicted and ground-truth
diacritized text files, which is a generic command that does not use our model necessarily.

### Training

To train the deep learning model, you need to execute the `train` command:

```bash
$ python multilevel_diacritizer/multi_level_diacritizer.py train --help
```

It has two mandatory arguments: `--train-data` (`-t`) and `--val-data` (`-v`).
They are used to specify the paths of the training and validation files, respectively.

The default model architecture has the following parameters:

- Window size: 150
- Sliding step: 30
- Embedding size: 128
- LSTM cells per layer in one direction: 128
- Dropout rate: 0.4
- Batch size: 1024

Each one of these parameters can be changed with the following arguments:
`--window-size`, `--sliding-step`, `--embedding-size`, `--lstm-size`,
`--dropout-rate`, `--batch-size`.

During the training a new folder called `params` will be generated in the working
directory if it does not exist (Its path can be altered by specifying the argument
`--params-dir`). Inside this folder, a model weights file will be generated and named
`MultiLevelDiacritizer-E#L#W#S#.h5`, where `#` indicates a number position. Each one
of the letters before the number positions indicates which hyperparameter the number
represents:

- `E`: Embedding size
- `L`: LSTM cells
- `W`: Window size
- `S`: Sliding step

According to the default values above, the default filename is `MultiLevelDiacritizer-E128L128W150S30.h5`.
If the file already exist before the start of the training, the train script will read that
file and assign the weights to the model before training. This is useful for stopping
the training and starting it again later without losing the current weights.

There are other parameters to specify other less important options, like the number of iterations and
the steps before early stopping. They can be found through the help of the `train` command that can be
accessed using the command written above.

### Diacritization

In addition to the web interface, the `diacritization` command can be used to predict the diacritics
of a raw Arabic text file.

```bash
$ python multilevel_diacritizer/multi_level_diacritizer.py diacritization --help
```

By default, the `diacritization` command takes the input text from the Standard Input Stream (STDIN)
and return the diacritized text to the Standard Output Stream (STDOUT). This behaviour can be changed
using the parameter `--file` (`-f`) to specify the path of the input file, and the parameter `--out-file`
(`-o`) to choose the path of the generated file.

```bash
$ python multilevel_diacritizer/multi_level_diacritizer.py diacritization -f input_file.txt -o output_file.txt
```

### Server

The `server` command is used to launch this tool in an interactive mode listening for HTTP requests. It
displays a web interface when it receives a GET request (for instance, through a browser), and it tries to
restore the diacritics of the plain text sent through a POST request (through the web app or directly from
a third party application).

```bash
$ python multilevel_diacritizer/multi_level_diacritizer.py server --help
```

If you have used the Docker method, the server starts running automatically when the container is launched.
Otherwise, it can be launched as indicated in the *Option 2* of the **Running** section above.


## Deployment

This web application contained in this projet is optimized for deployment on Heroku.
You can deploy it to your Heroku account by clicking the
following button (you can make an account if you do not have, it is free).

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/Hamza5/multilevel-diacritizer)
