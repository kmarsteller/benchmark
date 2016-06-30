**Instructions for running benchmark.py:**
==========================================

To start benchmarking, try:
`python benchmark.py [project] -[options]`
Results will be kept in a similarly-named database, e.g. `[project].db`

Commands and their output will be kept in a similarly-named log, e.g `logs/[project].log`

**SETTING UP A PROJECT TO DO BENCHMARKS**
------------------------------------------
Generally, create a `benchmarks` directory in your project's repository.  Within that, create files that start with `benchmark_`, e.g. `benchmark_item.py`.  Follow the general rules for a unit test, and create classes with functions that start with the prefix `benchmark_`.  If these naming conventions are followed, benchmarks will be found and run automatically.


**PROJECT-SPECIFIC JSON FILE (required)**:
------------------------------------------
`[project].json` will be read in if command `python benchmark.py [project]` is run and `[project].json` exists.  If `[project].json` does not exist, benchmark will fail.  Here are the JSON fields currently supported in that file:

`repository`:  the repo to be benchmarked.

`branch`:  the branch of that repo to benchmark.

`triggers`:  list of repositories, that if changed, should trigger a new benchmarking run.

`dependencies`: things needed by the `repository` in the conda env to successfully run the benchmark.

An example `[project].json` file:
```
{
    "repository": "git@github.com:user/repository",
    "branch":     "work",

    "triggers": [
        "git@github.com:user/trigger_repo_1",
        "git@github.com:user/trigger_repo_2",
        "https://bitbucket.org/user/trigger_repo_3"
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
}
```


**OPTIONS: (they're...optional)**
------------------------------------------
`--plot`, `-p`: To plot a specific spec from the database, after benchmarking has been run, use `python benchmark.py [project] --plot [spec]`

`--keep-env`, `-k`:  To keep your temporary conda environment around after benchmarking, usually for troubleshooting purposes.   *Note:* when your run ends, the env will be kept, but you'll be returned back to whatever env you started the run in. To inspect the kept env, do a `conda env list` and `source activate [envname]`

`--force`, `-f`: To force a run of the benchmarks even if no trigger repos have been changed, simply add `--force` to the command. This is usually used for testing, or if something went wrong with a previous run and you want the benchmarks to run again.

`--unit-tests`, `-u`: Runs the unit tests before running the benchmarks. Benchmarks running are contingent on unit tests passing.

`--dump`, `-d`: Dump the contents of the database to an SQL file.


**BENCHMARK.CFG FILE: (optional)**
------------------------------------------
`benchmark.cfg` will be used if it exists, but is generally only needed if administering a benchmark "server."

JSON file currently used to  specify fields such as:
`env`: set certain environment variables before use.

`slack`: post benchmarking results _message_ to _channel_ using _token_. (see JSON example below and Slack documentation for more details on custom integrations/hooks.)

`images`: can be used to _upload_ images to a specific _url_.
`data`: can be used to specify a place to _upload_ a database backup.

`working_dir` : where to look for everything.  defaults to where benchmark.py lives.

`repo_dir`:  where to clone the repositories of the main and trigger repos.  Defaults to `working_dir/repos`

`logs_dir`:  where to keep the logs for all of the projects being benchmarked.  Defaults to `working_dir/logs`

`remove_csv`:  remove benchmark data file after adding to database. Default: `False`.

`plot_history`: generate a plot showing history of each benchmark. Default: `True`.

`ca`:  CA cert information needed by curl, with two sub-data:

  `cacert`:  Defaults to "/etc/ssl/certs/ca-certificates.crt"
    `capath`:  Defaults to "/etc/ssl/certs"

`url`:  the base URL for linking to a benchmark website/URL handler running tornado_server.py

An example `benchmark.cfg`:
```
{
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

    "url" : "http://domain.com/benchmark/"
}
```

**RUNNING BENCHMARK ON A REGULAR BASIS**
------------------------------------------
It's suggested that if you want to run `benchmark` on a regular basis to keep track of history of code changes and how they change your performance, that you set `benchmark` to run on a regular schedule, for instance, in a `cron` job.

**COMPATIBILITY NOTE**
----------------------
Given some of the underlying tools used to make benchmark work, it will likely only work with Linux and MacOSX.
