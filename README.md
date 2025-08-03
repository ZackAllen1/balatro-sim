# Balatro Starting Strategy Simulation README

## Relevant Files
- `balatro_sim.py`: where the main simulation occurs, change the integer in the last line to alter the number of simulations run per strategy-difficulty combination
- `helper.py`: these are helper functions for calculating statistical significance and writing the actions taken for each strategy into an excel workbook called `balatro_simulation_data.xlsx`.

## Running the Program

Install the `pandas`, `numpy`, `simpy`, and `openpyxl` Python packages into your environment.
In the last line of `balatro_sim.py`, set the number of simulations you wish to run for
each strategy-difficulty combination (default = 500). Then run the `balatro_sim.py` file.
Once it is complete, an Excel file will be created called `balatro_simulation_data.xslsx` with every action played for each strategy. The terminal will print out the summary statistics for each strategy with 95% confidence intervals for average total score.
