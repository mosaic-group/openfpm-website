# OpenFPM / web

This repo contains the OpenFPM website [MkDocs](https://www.mkdocs.org/) sources. The project is inspired by the website for [MFEM](https://github.com/mfem/web) project. 

To make changes to the website you will need an install of Python version >= 3.6.9 and < 3.10 with the following libraries:

- use MkDocs v1.0.4 with Markdown v3.0 and the latest PyYAML and mkdocs-exclude-search:
  * `pip install --upgrade --user mkdocs==1.0.4`
  * `pip install --upgrade --user Markdown==3.0`
  * `pip install --upgrade --user PyYAML`
  * `pip install --upgrade --user mkdocs-exclude-search`
  * `pip install --upgrade --user "jinja2<3.1.0"`
- or in a Python virtual environment
  * `python3 -m venv openfpm-web-venv`
  * `. ./openfpm-web-venv/bin/activate`
  * `pip install -r requirements.txt`
- newer versions may not generate correct front page (to see the installed version, use `pip show mkdocs`)
- make sure you don't have `mkdocs-material` installed which may conflict with regular `mkdocs`
- clone this repo,
- edit or add some `.md` files (you may also need to update the `mkdocs.yml` config),
- preview locally with `mkdocs serve` (Windows users may need to specify a port, such as `mkdocs serve --dev-addr 127.0.0.1:4000`),
- publish with `mkdocs gh-deploy`.

