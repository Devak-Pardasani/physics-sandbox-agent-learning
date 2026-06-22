# Physics Sandbox Agent Learning

A small Python sandbox for collecting Newtonian motion episodes, training a simple
dynamics model, and replaying trajectories in a Pygame viewer.

The current learning pipeline focuses on 1D motion:

1. collect episodes with a random-force policy
2. train a PyTorch model to predict acceleration from recent observations
3. evaluate the trained checkpoint on held-out episodes
4. replay logged episodes with an optional model-prediction overlay

The code also includes reusable 1D/2D environment pieces and tests around the
simulation, logging, dataset construction, and replay loop.

## Requirements

- Python 3.11 or newer
- Pygame for replay rendering
- PyTorch for training and evaluation

Install the base dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the optional ML dependency when you want to train or evaluate models:

```bash
python -m pip install -r requirements-ml.txt
```

## Usage

Collect train, validation, and test episodes:

```bash
python main.py collect --mode 1d --episodes 200 --output-dir data/episodes/train
python main.py collect --mode 1d --episodes 50 --output-dir data/episodes/val
python main.py collect --mode 1d --episodes 50 --output-dir data/episodes/test
```

Train the acceleration model:

```bash
python main.py train \
  --train-dir data/episodes/train \
  --val-dir data/episodes/val
```

Evaluate the trained checkpoint:

```bash
python main.py evaluate \
  --checkpoint artifacts/models/dynamics_1d.pt \
  --test-dir data/episodes/test
```

Replay an episode:

```bash
python main.py render --episode data/episodes/test/<episode-file>.json
python main.py render \
  --episode data/episodes/test/<episode-file>.json \
  --checkpoint artifacts/models/dynamics_1d.pt
```

## Replay Controls

- `Space`: play or pause
- `Left` / `Right`: step backward or forward
- `R`: restart replay
- `Tab`: toggle debug text
- `M`: toggle mass visibility
- `Esc` or `Q`: quit

## Testing

Run the lightweight test suite:

```bash
PYGAME_HIDE_SUPPORT_PROMPT=1 python -m unittest discover -v
```

The PyTorch training smoke test is skipped automatically when PyTorch is not
installed. Install `requirements-ml.txt` to run that path locally.

## Generated Files

Episode logs, model checkpoints, evaluation summaries, exports, virtual
environments, and Python cache files are ignored by Git. Regenerate data with the
commands above instead of committing local runs.

## Project Layout

```text
main.py                  CLI entrypoint
config.py                Shared simulation configuration
agents/                  Policy interfaces and random policy
env/                     Sandbox environment and physics engines
models/                  Dataclasses for actions, states, observations, episodes
logging_tools/           Episode logging, loading, and export helpers
training/                Dataset building, model, checkpoint, and train loop
evaluation/              Model evaluation routine
ui/                      Pygame replay renderer and input handling
tests/                   Unit and smoke tests
```
