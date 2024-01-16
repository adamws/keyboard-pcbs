# How to

1. Create and activate virtual environment with access to `pcbnew` installed
together with KiCad

```shell
python -m venv --system-site-packages .env
. .env/bin/activate
```

2. Install dependencies

```shell
pip install -r requirements.txt
```

3. Run

```
python via_layouts_to_boards.py generate
python via_layouts_to_boards.py collect
```
