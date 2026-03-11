# Replication Code for 'Estimating Representative Causal Effects with Double Machine Learning'
> Provides code to replicate all tables and figures

## Suggested Workflow
- Clone this repository.
- Create a virtualenv by running `virtualenv venv` from the command line.
- Activate the virtualenv.  For example, on MacOS run `source venv/bin/activate`.
- Install the project dependencies by running `pip install -r requirements.txt`.
- Run `python code/simulation.py`

## Notes on Empirical Replication (Section 5)
- The `code/empirics.py` module is **not** based on the proprietary empirical dataset used in the paper.
- For confidentiality reasons, the actual Section 5 dataset is not included in this repository.
- The script is provided as runnable example code that can operate on:
  - simulated data (default), or
  - user-supplied real data via `run_rorr(data=...)`.
- As a result, the empirical figures and `figures/table-3-section5.tex` generated from this repository do **not** match the Section 5 empirical results reported in the paper.
