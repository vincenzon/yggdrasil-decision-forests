#! /bin/bash
sed -i '/PS1=/ s/u@\\h/u/' ~/.bashrc

if ! command -v pipx &> /dev/null; then
  pip install --user pipx
  pipx ensurepath
fi

if ! command -v uv &> /dev/null; then
  pipx install uv
fi

if [ ! -f pyproject.toml ]; then
  uv init --bare
  uv venv
  uv pip install numpy polars ipykernel flake8
fi

# Install documentation dependencies
uv pip install ydf mkdocs mkdocs-material mkdocs-autorefs mkdocs-jupyter \
  mkdocs-glightbox mkdocs-macros-plugin mkdocs-gen-files \
  mkdocstrings[python] mkdocstrings-crystal black griffe_inherited_docstrings pandas

