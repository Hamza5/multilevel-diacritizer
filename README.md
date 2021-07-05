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

## Deployment

This repository contains a small web application optimized for deployment on Heroku.
It gives the user a simple web interface to input any text, then send it to the backend
to be diacritized by our deep learning model, then the result will be displayed
on the web interface. You can deploy it to your Heroku account by clicking the
following button (you can make an account if you do not have, it is free).

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)