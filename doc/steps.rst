Estimation Steps
-----------------

1. Run `python/data_moments_weighting.py`: This module extracts moments from the SOEP data.
Based on this observed moments, a weighting matrix for the SMM routine is calculated
in a bootstrap procedure. The lists of moments and corresponding weights are saved
in the `estimation/basecamp/resources` directory.

2. Run `estimation/basecamp/run.py` to estimate free parameters of the model in an SMM routine
based on SOEP moments.

3. Run `python/create_report.py` to create figures describing the model fit given the
estimated parameter vector in `figures`.

