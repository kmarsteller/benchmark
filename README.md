Instructions for running benchmark.py:


To start benchmarking, try:
`python benchmark.py [project] --options`
Results will be kept in a similarly-named database, e.g. `[project].db`
commands and their output will be kept in a similarly-named log, e.g `logs/[project].log`

*PROJECT-SPECIFIC JSON FILE (required)*:  [project].json will be read in if command `python benchmark.py [project]` is run and [project].json exists.
`repository`:  the repo to be benchmarked.
`branch`:  the branch of that repo to benchmark.
`triggers`:  list ofrepositories, that if changed, should trigger a new benchmarking run.
`dependencies`:  other things needed by the repository to be installed in the conda env in order to successfully run the benchmark.

An example .json file looks like this:
```
{
    "repository": "git@github.com:user/repository",
    "branch":     "work",

    "triggers": [
        "git@github.com:openmdao/OpenMDAO",
        "git@github.com:openmdao/MBI",
        "https://bitbucket.org/mdolab/pyoptsparse"
    ],

    "dependencies": [
        "python=2.7",
        "numpy",
        "scipy",
        "networkx",
        "sqlitedict",
        "mpi4py",
        "git+https://bitbucket.org/petsc/petsc@v3.5",
        "git+https://bitbucket.org/petsc/petsc4py@3.5"
    ]
}```


*OPTIONS: (optional)*
`--plot`: To plot a specific spec from the database, after benchmarking has been run, use `python benchmark.py project --plot [spec]`

`--keep-env`:	To keep your temporary conda environment around after benchmarking,
(for troubleshooting purposes),  Note: when your run ends, the env will be kept, but you'll be returned back to whatever env you started the run in. To inspect the kept env, do a `conda env list` and `source activate envname`

`--force`: To force a run of the benchmarks even if no trigger repos have been changed, simply add `--force` to the command. This is usually used for testing, or if something went wrong with a previous run and you want the benchmarks to run again.

`--unit-tests`: Runs the unit tests before running the benchmarks.  If the unit tests fail, benchmarks will not be run.

`--dump`: Dump the contents of the database to an SQL file


*BENCHMARK.CFG FILE: (optional)*  
`benchmark.cfg` will be used if it exists, generally only needed if adminning a benchmark "server."
JSON file currently used to  specify fields such as:
`env`: set certain environment variables before use.
`slack`: post benchmarking results _message_ to _channel_ using _token_. (see Slack documentation for more details on custom integrations/hooks.)
`images`: can be used to _upload_ images to a specific _url_.
`data`: can be used to specify a place to _upload_ a database backup.

An example `benchmark.cfg` might look like this:
```{
    "env": {
        "LD_PRELOAD": "/path/to/libgfortran.so.3"
    },

    "slack": {
        "message": "https://hooks.slack.com/services/[hook]/[address]",
        "file": "https://hooks.slack.com/services/[hook]/[address]",
        "channel": "#benchmarks",
        "token":   "slack-token-serial-number"
    },

    "images": {
        "url": "http://domain.com/path/to/images",
        "upload": "user@server.domain.com:directory/subdir"
    },

    "data": {
    	"upload": "user@server.domain.com:directory"
    }
}```


