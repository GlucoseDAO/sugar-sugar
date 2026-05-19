# sugar-sugar

A game to test your glucose-predicting superpowers! 🎯

> 🎵 The name "sugar-sugar" was inspired by a scientific remake of The Archies' classic hit song ["Sugar, Sugar"](https://www.youtube.com/watch?v=jJvAL-iiLnQ) from 1969!

## What is Sugar Sugar?

Sugar Sugar is a web-based research app built with [Plotly Dash](https://dash.plotly.com/). It turns glucose forecasting into an interactive game: you see part of a CGM trace, draw how you think it will continue, then compare your prediction with the real values. Your accuracy is measured with standard forecasting metrics (MAE, RMSE, MAPE).

## Why this study matters

### For people with diabetes

Accurate short-term glucose forecasts can support safer day-to-day decisions about insulin, carbohydrates, exercise, and timing. Better prediction can help reduce dangerous highs and lows and make glucose management less stressful.

### For people without diabetes

Glucose is a core metabolic signal linked to energy, cognition, mood, exercise response, and long-term health. Prediction can reveal how food, sleep, stress, and activity shape metabolism in real time — useful for preventive health and lifestyle optimisation.

### Establishing a human baseline for machine learning

There is currently no published human-baseline benchmark for CGM glucose prediction. Without knowing how well an informed human can predict glucose trends, we cannot assess whether any ML model is genuinely useful — it might be worse than a knowledgeable diabetic, or better only in specific scenarios. This study establishes that baseline so future ML models can be compared meaningfully.

For a deep dive into state-of-the-art ML approaches to glucose prediction, see [GlucoBench](https://github.com/IrinaStatsLab/GlucoBench) ([paper](https://arxiv.org/abs/2410.05780)).

## Ethics approval and study credibility

Sugar Sugar is part of a real research effort. The study received clearance from the Ethics Committee of University Medicine Rostock (Ethikkommission der Universitätsmedizin Rostock), Germany. It is an open-source, community-driven project run by [GlucoseDAO](https://github.com/GlucoseDAO).

## How the gameplay works

1. **Load your data** — Upload a Dexcom, Libre, Medtronic, or Nightscout file, or use the built-in generic sample dataset.
2. **Make predictions** — Click and drag on the glucose chart to draw your forecast for the hidden hour ahead.
3. **Compare results** — Your predictions are compared against the actual glucose values.
4. **See your accuracy** — Get MAE, RMSE, and MAPE metrics showing how close you were.

The game has up to 12 rounds. Each round takes about 2–3 minutes; a full session is around 30 minutes. You can stop at any time and resume later in the same browser.

## Screenshots

![Game Interface](images/screenshot.png)
*sugar-sugar in action — try to predict where that line is going!*

## Data and privacy

Sugar Sugar supports two modes:

- **Play-only mode**: check **"I just want to play (do not store my CGM / gameplay data)"** on the landing page and the app will not save any study outputs. No data leaves your device.
- **Study mode** (default): if you do not check play-only, the app may write **consent + study outputs** to CSV for research (predictions vs ground truth, accuracy metrics, anonymised questionnaire responses).

Until you choose to submit, everything stays in your own browser's localStorage. Nothing is sent to our servers without your active consent.

Raw uploaded CGM files are used only to run the game and are not kept permanently.

## Resume and study integrity

Your progress is saved locally in the browser. You can close the tab and return later — you will be offered to continue from where you left off.

Please do not restart just because you are unhappy with your accuracy. People naturally prefer to keep good scores and discard bad ones, but that would skew the study and make the results less scientifically valid. If you feel disappointed, continue with another round rather than resetting your progress.

## Bug reports and contributing

### Found a bug?

Please open an issue at [https://github.com/GlucoseDAO/sugar-sugar](https://github.com/GlucoseDAO/sugar-sugar). Bug reports are very helpful for improving the app and for catching anything that could affect data quality.

### Software developers

Contributions are welcome through the project GitHub. We especially welcome:

- Bug fixes and UI improvements
- Support for more CGM devices or file formats
- Localisation improvements

Because this is an active study, changes that affect the core study design (new questionnaire fields, modifications to the prediction task) should be discussed with the team first to ensure data consistency across participants.

### Everyone else

You can still help a lot:

- Share the study with CGM communities, diabetes support groups, and metabolic health enthusiasts.
- Report unclear wording or usability issues via GitHub issues.
- Reach out through the [Contact](https://github.com/GlucoseDAO/sugar-sugar) page if you would like to collaborate.

## Contribution statement

- **Livia Zaharaia** (GlucoseDAO) — Core Developer
- **Anton Kulaga** (Institute for Biostatistics and Informatics in Medicine and Ageing Research) — Core Developer
- **Irina Gaynanova** (Department of Statistics and Department of Biostatistics, University of Michigan) — Scientific Advisor

## Technical setup

### Prerequisites

- Python 3.11 or higher
- UV (Python package manager)

### Installing UV

```bash
# Windows/macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew (macOS)
brew install uv
```

For full installation instructions see the [UV documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Installation

```bash
git clone https://github.com/GlucoseDAO/sugar-sugar.git
cd sugar-sugar
uv sync
```

### Chrome / Chromium (for share-card image export)

The share page renders a PNG card using [kaleido](https://github.com/plotly/Kaleido), which needs a Chromium-based browser. On first startup the app checks for an installed browser (Chrome, Chromium, Brave, Edge); if none is found it automatically downloads **Chrome for Testing** (~130 MB) into `~/.local/share/choreographer/deps/`.

Most developer machines already have Chrome, so nothing extra is needed. For headless servers or Docker, either:

```bash
# Option A: install system chromium (Debian/Ubuntu)
apt-get install -y chromium

# Option B: pre-download before first run
uv run setup-chrome
```

Both options coexist safely — the app always checks the local download first, then falls back to a system browser.

### Docker

```bash
docker build -t sugar-sugar .
docker run -p 8050:8050 sugar-sugar
```

The included `Dockerfile` installs system Chromium so share-card rendering works out of the box.

### Configuration

```bash
cp .env.template .env
```

Edit `.env` to configure your server settings:

```bash
DASH_HOST=127.0.0.1
DASH_PORT=8050
DASH_DEBUG=True
DEBUG_MODE=False
UMAMI_SCRIPT_URL=https://sugar-sugar.glucosedao.org/stats/script.js
UMAMI_WEBSITE_ID=7c6fb178-d8ff-439e-a9f3-e289d9ec7e97
UMAMI_DOMAINS=sugar-sugar.glucosedao.org
UMAMI_HOST_URL=https://sugar-sugar.glucosedao.org/stats
```

Umami tracking is disabled when either `UMAMI_SCRIPT_URL` or `UMAMI_WEBSITE_ID` is blank.
`UMAMI_DOMAINS` limits counting to listed production hostnames, and `UMAMI_HOST_URL`
points Umami pageview requests at the same-domain `/stats` proxy.

### Running the app

```bash
uv run start
```

Override via command line:

```bash
uv run start --host 0.0.0.0 --port 3000 --debug
```

### Quick chart debugging

Skip landing/startup/consent and jump straight to the prediction chart:

```bash
uv run chart
uv run chart --file /path/to/export.csv
uv run chart --prefill                # pre-fill predictions for testing submit flow
uv run chart --prefill --noise 0.10   # ±10% noise
uv run chart --unit mmol/L --locale de
```

### Clearing localStorage during development

```bash
uv run start --clean   # clears localStorage on every page load
```

### Testing the resume dialog

1. `uv run start`
2. Walk through landing and startup pages.
3. Close the tab (do **not** stop the server).
4. Reopen `http://127.0.0.1:8050/` — the resume dialog should appear.

## Technical architecture

```
sugar_sugar/
├── app.py                  # Main application and routing logic
├── config.py               # Application constants and configuration
├── data.py                 # Data loading and processing utilities
├── consent.py              # Consent CSV persistence utilities
└── components/
    ├── landing.py          # Landing + consent/choices page
    ├── startup.py          # User registration (demographics/medical history)
    ├── header.py           # Game controls and file upload
    ├── glucose.py          # Interactive glucose visualization
    ├── predictions.py      # Prediction data table
    ├── metrics.py          # Accuracy metrics display
    ├── submit.py           # Game submission logic
    └── ending.py           # Results summary page
```

State management uses Dash `dcc.Store` components. The storage backend is controlled by `STORAGE_TYPE`:

| Value | Behaviour |
|-------|-----------|
| `local` (default) | Persists in `localStorage` — survives browser restarts. |
| `session` | Persists in `sessionStorage` — cleared when the tab closes. |
| `memory` | Lives only in React state — cleared on any page refresh. |

## Known issues

- Nightscout import is planned but not yet fully implemented.
- No scoring system or difficulty levels yet.
