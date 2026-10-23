# Gomoku-Zero
Inspired by the google AlphaGo project **AlphaZero-style Gomoku AI** built with Python and PyTorch. The project includes a Gomoku environment, a policy/value neural network, Monte Carlo Tree Search (MCTS), self-play training, terminal play, and an optional Playwright adapter for playing on PaperGames.io.

> **Project status:** This is a learning and experimentation project, it develops a self-learning rebot to play Gomuku game as like AlphaGo. The trained model currently is not very smart enough to win the online-bot. However it understands the board and rules of winning.
## Features

- **Gomoku environment:** board state, legal moves, turn handling, and five-in-a-row win detection.
- **AlphaZero-style learning:** a convolutional policy/value network trained on self-play positions.
- **MCTS:** uses the network's policy priors and value estimates to select moves.
- **Self-play trainer:** configurable board size, model width/depth, search simulations, iterations, and training updates.
- **Terminal gameplay:** play against a locally trained model.
- **PaperGames.io adapter (`play_online.py`):** reads the board from the browser, waits for the opponent's moves, supports manual or automatic AI turns, checks immediate wins and single-point blocks, and prints the final board when it detects a result.

## Requirements

- Python 3.10 or newer recommended.
- PyTorch and NumPy.
- Playwright and its Chromium browser for online play.
- On Apple Silicon Macs, PyTorch can use the MPS device when it is available. CPU mode is also supported.

## Installation

Clone your repository and enter the project directory:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Gomoku-Zero
```

Create and activate a virtual environment on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To use the online browser adapter, install Playwright and Chromium:

```bash
python -m pip install playwright
python -m playwright install chromium
```

Check the PyTorch installation and whether MPS is available on your Mac:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

If your environment does not support MPS, use `--device cpu` for training or local play. For online play, the script's device option can also be set to `cpu`.

## Train a model

The training script starts with randomly initialized weights unless you provide a checkpoint to resume from. A larger search/training run can be computationally expensive, especially on a 15×15 board.

Example: train a 15×15 checkpoint for use with the 15×15 PaperGames board:

```bash
python train.py \
  --size 15 \
  --channels 64 \
  --blocks 3 \
  --simulations 100 \
  --games-per-iteration 20 \
  --iterations 100 \
  --updates 30 \
  --batch-size 64 \
  --device auto \
  --output checkpoints/15x15_tactics.pt
```

Adjust the number of simulations, games, and iterations to match your hardware and available time. A model trained on a different board size cannot be used as a drop-in replacement for a 15×15 checkpoint; the network's policy head depends on the board dimensions.

To resume from a checkpoint:

```bash
python train.py \
  --resume checkpoints/15x15_tactics.pt \
  --output checkpoints/15x15_tactics.pt \
  --device auto
```

Resuming restores the model and optimizer state when available, but the replay buffer is not persisted between runs.

## Play locally in the terminal

Use a checkpoint trained for the board size you want to play:

```bash
python play.py --model checkpoints/15x15_tactics.pt --simulations 100
```

Choose the human's color and the device explicitly if needed:

```bash
python play.py \
  --model checkpoints/15x15_tactics.pt \
  --simulations 100 \
  --human white \
  --device cpu
```

## Play on PaperGames.io

`play_online.py` is an optional browser automation script. It opens PaperGames.io in Chromium and uses Playwright to read the board and click the AI's selected cell.

1. Make sure you have a compatible **15×15 model checkpoint**. By default, the script looks for `checkpoints/15x15_tactics.pt`.
2. Install Playwright and Chromium using the commands above.
3. Run the script from the project root:

   ```bash
   python play_online.py
   ```

4. Join or start a Gomoku game in the opened browser, then press Enter in the terminal when prompted.
5. The script attempts to detect the local player's color from the player panel. If it cannot, it asks whether the AI should play Black or White.
6. By default, press Enter on each AI turn to calculate and place a move. To calculate automatically on every AI turn, use:

   ```bash
   python play_online.py --auto
   ```

Useful options:

```bash
python play_online.py --model checkpoints/15x15_tactics.pt --simulations 100 --device mps
python play_online.py --model checkpoints/15x15_tactics.pt --simulations 100 --device cpu
```

The adapter prioritizes an immediate AI win, then blocks a single immediate opponent winning point, and otherwise uses MCTS. It cannot guarantee a win or defend against every multi-threat position. It relies on PaperGames.io's current DOM selectors and may need maintenance if the website changes. Please supervise online games and confirm that moves are accepted by the website.

When a win or draw is detected from the board, the script prints the result and final board, then closes the browser. If the script is interrupted or encounters an error, the browser is also closed by the current cleanup handler.

## Tests

Run the unit tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The online adapter requires an active browser session and is not covered by ordinary unit tests; verify it manually against the current website.

## Project layout

```text
Gomoku-Zero/
├── alphazero/
│   ├── mcts.py            # Monte Carlo Tree Search
│   ├── network.py         # Policy/value neural network
│   └── training.py        # Self-play, updates, and checkpoints
├── gomoku/
│   ├── __init__.py
│   └── environment.py     # Gomoku rules and game state
├── tests/                 # Unit tests
├── checkpoints/           # Local model files (not committed by default)
├── main.py                # Original baseline entry point
├── play.py                # Terminal human-vs-AI game
├── play_online.py         # Optional PaperGames.io browser adapter
├── train.py               # Training entry point
├── requirements.txt
└── README.md
```

## Checkpoints and GitHub

Model checkpoints are ignored by the repository's `.gitignore` (`checkpoints/*.pt`) because they can be large. A fresh clone will not automatically contain `checkpoints/15x15_tactics.pt`; train a compatible model locally or distribute the checkpoint separately if you want others to run the online adapter immediately.

Do not commit your virtual environment, Python cache files, credentials, or other local-only artifacts.

## How the AlphaZero-style loop works

1. The network estimates a policy over board positions and a value for the current position.
2. MCTS uses those estimates while exploring candidate moves.
3. Self-play games generate move-policy targets and final outcomes.
4. Training updates the policy head from the search targets and the value head from game results.

This implementation is deliberately compact and is intended to support experimentation and learning rather than reproduce every optimization in DeepMind's original AlphaZero system.
