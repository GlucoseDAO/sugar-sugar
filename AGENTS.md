## Project overview

This project is a Sugar-Sugar game where the user gets the glucose value for some timespan and has to predict by drawing the lines on the chart. He is given a sequence that he has to prolong. The aim of the study is to measure human accuracy of the glucose predictions.
The project is a DASH app, with app.py being main, while glucose, metrics, prediction and startup are dash components. I has default example csv file to play and debug with but provide an option to upload your own csv files from dexcom, libre and other CGM-s.
We use session storage to allow multiple users workin on the same app. Predictions are stored in polars dataframe, there is also a dataframe for current prediction window and scrolling positions.
When the user draws the line it interpolates the position to detect closes glucose and time value (time measurements are done every 5 minutes) and then updates the dataframe with the prediction values.

## Build and test commands

uv is used as the package manager for the project.
uv run start is used to run the dash app.

## Code style guidelines

Always use type-hints. For file pathes prefer to use pathlib, for cli - typer, for dataframes - polars. We try to split logic into components and use functional style when possible, avoiding unneccesary mutability and duplication.