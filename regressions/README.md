hvm_dataset_class is just a simple class for extracting the hvm data formatted usefully for running regressions -- most of it is from the Jupyter notebooks from Dan's course.

regression_class is the class that runs the regressions -- note that currently it is set up to predict individual units rather than fit to the entire population of units (ie. each unit has its on regularization coefficient). 

run_ind_unit_regression is a script that creates regression objects and runs the regression

run.ind_unit_regression is a slurm script to submit slurm jobs to run the regressions

-obviously fitting a unique regression coefficient takes more compute/time to run the regressions so there is a trade off there.

