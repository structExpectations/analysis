Estimation Steps
-----------------

1. Run `data_moments_weighting.py`: This module extracts moments from the SOEP data.
Based on this observed moments, a weighting matrix for the SMM routine is calculated
in a bootstrap procedure. The lists of moments and corresponding weights are saved
in the `basecamp/resources` directory.

2.