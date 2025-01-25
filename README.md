# sugar-sugar
A fun game to test your glucose-predicting superpowers! 🎯

Ever wondered how good you are at predicting where your glucose levels are heading? sugar-sugar turns this into an engaging game where you can test your glucose-spotting skills. Look at your CGM data, make your best guess about future values, and see how close you get!

Why did we create this? While fancy AI models are being built to predict glucose values, we realized nobody really knows how good humans are at this - especially experienced CGM users who've developed an instinct for their patterns. By playing sugar-sugar, you're not just having fun, you're also helping us understand:
- How accurate are humans at predicting glucose trends?
- What patterns do experienced CGM users notice that computers might miss?
- Could this help make better prediction tools in the future?

> 🎵 Fun fact: The name "sugar-sugar" was inspired by a scientific remake of The Archies' classic hit song ["Sugar, Sugar"](https://www.youtube.com/watch?v=jJvAL-iiLnQ) from 1969!

## Screenshots
![Game Interface](images/screenshot.png)
*sugar-sugar in action - try to predict where that line is going!*

## Setup

### Prerequisites
- Python 3.10 or higher
- Poetry (Python package manager)

### Installing Poetry
1. **Windows**:
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

2. **macOS/Linux**:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. **Alternative method** (using pip):
```bash
pip install poetry
```

For detailed installation instructions, visit the [Poetry documentation](https://python-poetry.org/docs/#installation).

### Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/sugar-sugar.git
cd sugar-sugar
```

2. Install dependencies using Poetry:
```bash
poetry install
```

### Running the Game
```bash
poetry run python -m sugar_sugar.app
```

## KNOWN ISSUES:

- For the ease of prototyping we used ugly global variables, so at the moment the game does not support multiple users. We will switch to session-based state management soon.
- Currently only Dexcom and Libre 3 are supported. We will add support for other CGM devices soon.
- No scoring system and difficulty levels yet.

## FAQ

### Is this production-ready software?
This is an early-stage project meant for research and experimentation. So far it is single-user only but we will fix it soon

### Do you use my personal data?
No, we only use the data you upload to allow you play the game. We do not store any data from your uploads (the data is loaded to temp folder and deleted after the game session is over).

### How accurate can glucose predictions be?
Glucose prediction is complex! Research shows that CGM data alone often isn't enough for highly accurate predictions. Other factors like physical activity, meals, insulin, and stress play crucial roles. For a deep dive into state-of-the-art machine learning approaches to glucose prediction, check out [GlucoBench](https://github.com/IrinaStatsLab/GlucoBench) ([paper](https://arxiv.org/abs/2410.05780)), which provides benchmarks and datasets for glucose prediction models.

### Can I contribute?
We welcome pull requests, bug reports, and feature suggestions through GitHub issues. Check out our contributing guidelines for more details.

### I have an idea for improvement!
Great! Feel free to:
- Open an issue to discuss your idea
- Submit a pull request with your changes
- Reach out to the contributors directly

## Contribution statement
- **Livia Zaharaia** (GlucoseDAO) - Core Developer
- **Anton Kulaga** (Institute for Biostatistics and Informatics in Medicine and Ageing Research) - Core Developer
- **Irina Gaynanova** (Department of Statistics and Department of Biostatistics, University of Michigan) - Scientific Advisor

## Technical Architecture

sugar-sugar is built with [Plotly Dash](https://dash.plotly.com/), creating an interactive web app that allows you to view CGM data, make predictions, and analyze accuracy. The app supports both Dexcom G6 and Libre 3 data formats, processing them automatically without storing any personal information.

### Main Components

1. **Glucose Chart**: Interactive visualization showing your glucose data, color-coded ranges, event markers, and predictions

2. **Metrics Display**: Real-time accuracy measures for your predictions

3. **Prediction Table**: Tracks predicted values, actual values, and prediction errors

4. **Header Controls**: Upload data, adjust time windows, and customize display settings