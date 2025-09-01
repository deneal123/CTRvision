
# Install 
> *I checked with cuda vers 12.2 for local start up*

1. In project directory
```bash
git clone https://github.com/deneal123/CTRvision.git
```

2. For install dependencies
```bash
bash install.sh
```

*Warning, if you still can't run scripts, should follow:*
```bash
rm -rf ./.venv
pyenv init # follows instructions
pyenv shell 3.10.6
uv cache clean
uv python pin 3.10.6
uv sync
```
*Make sure you are using the actual version of Python 3.10.6.*

## Local

```bash
bash notebook.sh
```

## Docker

```bash
bash build.sh
```

```bash
bash run.sh
```
