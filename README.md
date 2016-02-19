Instructions for running benchmark.py:


To start benchmarking, try:
"python benchmark.py project"
where project is the name of the json configuration file that you want to run, e.g.
"python benchmark.py openmdao" to benchmark using "openmdao.json"
results will be kept in a similarly-named database, e.g. "openmdao.db"
commands and their output will be kept in a similarly-named log, e.g "openmdao.log"


OPTIONS:
________

To plot a specific spec from the database, after benchmarking has been run, use:
"python benchmark.py project --plot spec"

To keep your temporary conda environment around after benchmarking,
(for troubleshooting purposes), try (for example):
"python benchmark.py openmdao --keep-env:
Note: when your run ends, the env will be kept, but you'll be returned back
to whatever env you started the run in.
To inspect the kept env, do a "conda env list"
and "source activate envname"

To force a run of the benchmarks even if no trigger repos have been changed,
simply add --force to the command, e.g. "python benchmark.py openmdao --force"


CFG FILE:
_________
working_dir:  the directory in which benchmark.py and its resulting files will live.
remove_csv:  whether or not to remove the comma separated values file from a local directory


JSON FILE:
__________
repository:  the repo to be benchmarked
branch:  the branch of that repo to benchmark
triggers:  repositories, that if changed, should trigger a new benchmark
dependencies:  other things needed by the repository in order to run the benchmark