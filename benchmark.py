#!/usr/bin/env python
from __future__ import print_function

from subprocess import Popen, PIPE

import sys
import os
import shlex
import sqlite3
import time
import json
import csv

from argparse import ArgumentParser

from contextlib import contextmanager

benchmark_dir = os.path.abspath(os.path.dirname(__file__))

# default configuration options
conf = {
    "working_dir": benchmark_dir,
    "remove_csv":  False
}

import logging
logging.basicConfig(filename='benchmark.log',level=logging.DEBUG)


class BenchmarkDatabase(object):
    def __init__(self, name):
        self.dbname = name+".db"
        self.conn   = sqlite3.connect(self.dbname)
        self.cursor = self.conn.cursor()

    def _ensure_commits(self):
        """
        if the commit tables have not been created yet, create them
        """
        # a table containing the last benchmarked commit for each trigger
        # repository
        self.cursor.execute("CREATE TABLE if not exists LastCommits"
                            " (Trigger TEXT UNIQUE, LastCommitID TEXT)")

        # a table containing the commit ID for each trigger repository
        # for a given benchmark run (specified by DateTime)
        self.cursor.execute("CREATE TABLE if not exists Commits"
                            " (DateTime INT, Trigger TEXT, CommitID TEXT,"
                            "  PRIMARY KEY (DateTime, Trigger))")

    def _ensure_benchmark_data(self):
        """
        if the bechmark data table has not been created yet, create it
        """
        self.cursor.execute("CREATE TABLE if not exists BenchmarkData"
                            " (DateTime INT, Spec TEXT, Status TEXT, Elapsed REAL, Memory REAL,"
                            "  PRIMARY KEY (DateTime, Spec))")

    def get_last_commit(self, trigger):
        """
        Check the database for the most recent commit that was benchmarked
        for the trigger repository
        """
        self._ensure_commits()

        self.cursor.execute("SELECT LastCommitID FROM LastCommits WHERE Trigger == ?", (trigger,))
        rows = self.cursor.fetchall()
        if rows:
            return rows[0][0]
        else:
            return ''

    def update_commits(self, commits, timestamp):
        """
        update commits
        """
        self._ensure_commits()

        for trigger, commit in commits.items():
            logging.info('INSERTING COMMIT %s %s' % (trigger, commit))
            self.cursor.execute('INSERT OR REPLACE INTO LastCommits VALUES (?, ?)', (trigger, str(commit)))
            self.cursor.execute('INSERT INTO Commits VALUES (?, ?, ?)', (timestamp, trigger, str(commit)))

    def add_benchmark_data(self, commits, filename):
        """
        Insert benchmarks results into BenchmarkData table.
        Create the table if it doesn't already exist.
        """
        self._ensure_benchmark_data()

        data_added = False

        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                spec = row[1].rsplit(':', 1)[1]
                logging.info('INSERTING BenchmarkData %s' % str(row))
                self.cursor.execute("INSERT INTO BenchmarkData VALUES(?, ?, ?, ?, ?)", (row[0], spec, row[2], float(row[3]), float(row[4])))
                data_added = True

        if data_added:
            self.update_commits(commits, row[0])  # row[0] is the timestamp for this set of benchmark data

    def dump_benchmark_data(self):
        with open(self.dbname+'.sql', 'w') as f:
            for line in self.conn.iterdump():
                f.write('%s\n' % line)


def read_json(filename):
    """
    read data from a JSON file
    """
    with open(filename) as json_file:
        return json.loads(json_file.read())


def get_exitcode_stdout_stderr(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """
    print(cmd)
    logging.info("CMD => %s" % cmd)
    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    rc = proc.returncode
    logging.info("RC => %d" % rc)
    if out:
        logging.debug("STDOUT =>\n%s" % out)
    if err:
        logging.debug("STDERR =>\n%s" % err)
    return rc, out, err


@contextmanager
def cd(newdir):
    """
    A cd that will better handle error and return to its orig dir.
    """
    logging.info('cd into %s' % newdir)
    prevdir = os.getcwd()
    fulldir = os.path.expanduser(newdir)
    if not os.path.exists(fulldir):
        os.makedirs(fulldir)
    os.chdir(fulldir)
    try:
        yield
    finally:
        logging.info('cd from %s back to %s' % (fulldir, prevdir))
        os.chdir(prevdir)


@contextmanager
def repo(repository, repo_dir, branch=None):
    """
    cd into local copy of repository.  if the repository has not been
    cloned yet, then clone it to working directory first.
    """
    prev_dir = os.getcwd()

    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)
    logging.info('cd into repo dir %s from  %s' % (repo_dir, prev_dir))
    os.chdir(repo_dir)

    repo_name = repository.split('/')[-1]
    if not os.path.isdir(repo_name):
        clone_repo(repository, branch)
    else:
        # TODO: could possibly be there but wrong branch?
        pass

    logging.info('cd into repo %s' % repo_name)
    print('cd into repo %s' % repo_name)
    os.chdir(repo_name)
    try:
        yield
    finally:
        logging.info('cd from repo %s back to %s' % (repo_name, prev_dir))
        os.chdir(prev_dir)


def benchmark(project_info, force=False, keep_env=False):
    current_commits = {}
    update_triggered_by = []

    db = BenchmarkDatabase(project_info["name"])
    
    #remove previous repo_dirs and clone fresh ones to avoid trouble.
    repo_dir= os.path.expanduser(os.path.join(conf["working_dir"], (project_info["name"] + "_repos")))
    remove_repo_dir(repo_dir)
    
    if force:
        update_triggered_by.append('force')
    else:
        triggers = project_info["triggers"]
        triggers.append(project_info["repository"])

        for trigger in triggers:
            # for the project repository, we may want a particular branch
            if trigger is project_info["repository"]:
                branch = project_info.get("branch", None)
            else:
                branch = None
            # check each trigger for any update since last run
            with repo(trigger, repo_dir, branch):
                print('checking trigger', trigger, branch if branch else '')
                last_commit = str(db.get_last_commit(trigger))
                logging.info("Last CommitID: %s" % last_commit)
                current_commits[trigger] = get_current_commit()
                logging.info("Current CommitID: %s" % current_commits[trigger])
                if (last_commit != current_commits[trigger]):
                    logging.info("There has been an update to %s\n" % trigger)
                    print("There has been an update to %s" % trigger)
                    update_triggered_by.append(trigger)

    if update_triggered_by:
        logging.info("Benchmark triggered by updates to: %s" % str(update_triggered_by))
        print("Benchmark triggered by updates to: %s" % str(update_triggered_by))
        env_name = create_env(project_info["name"])
        activate_env(env_name, project_info["triggers"],
                               project_info.get("dependencies", []),
                               repo_dir)
        with repo(project_info["repository"], repo_name, project_info.get("branch", None)):
            get_exitcode_stdout_stderr("pip install -e .")
            csv_file = env_name+".csv"
            run_benchmarks(csv_file)
            db.add_benchmark_data(current_commits, csv_file)
            if conf["remove_csv"]:
                os.remove(csv_file)

        db.dump_benchmark_data()
        remove_env(env_name, keep_env)

def clone_repo(repository, branch):
    """
    clone repository into current directory
    """
    if branch:
        git_clone_cmd = "git clone -b %s --single-branch %s" % (branch, repository)
        hg_clone_cmd = "hg clone %s -r %s" % (repository, branch)
    else:
        git_clone_cmd = "git clone " + repository
        hg_clone_cmd = "hg clone " + repository

    code, out, err = get_exitcode_stdout_stderr(git_clone_cmd)
    if code:
        code, out, err = get_exitcode_stdout_stderr(hg_clone_cmd)
    if code:
        raise RuntimeError("Could not clone %s" % repository)


def get_current_commit():
    """
    Update and check the current repo for the most recent commit.
    """
    pull_git = "git pull"
    pull_hg = "hg pull; hg merge"

    commit_git = "git rev-parse HEAD"
    commit_hg = "hg id -i"

    # pull latest commit from desired branch and get the commit ID
    code, out, err = get_exitcode_stdout_stderr(pull_git)
    if (code is 0):
        code, out, err = get_exitcode_stdout_stderr(commit_git)
    else:
        code, out, err = get_exitcode_stdout_stderr(pull_hg)
        code, out, err = get_exitcode_stdout_stderr(commit_hg)

    return out


def create_env(project):
    """
    Create a conda env.
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    env_name = project + "_" + timestr
    conda_create = "conda create -y -n " + env_name + " python=2.7 pip numpy scipy swig psutil"
    code, out, err = get_exitcode_stdout_stderr(conda_create)
    if (code == 0):
        return env_name
    else:
        raise RuntimeError("Failed to create conda environment", env_name, code, out, err)


def activate_env(env_name, triggers, dependencies, repo_name):
    """
    Activate an existing conda env and install triggers and dependencies into it

    Triggers are installed from a local copy of the repo using setup.py install

    Dependencies are pip installed
    """
    # activate environment by modifying PATH
    logging.info("PATH AT FIRST: %s" % os.environ["PATH"])
    path = os.environ["PATH"].split(os.pathsep)
    os.environ["KEEP_PATH"] = path[0]  # old path leader
    path.remove(path[0])  # remove default conda env
    path = (os.pathsep).join(path)
    logging.info("env_name: %s, path: %s" % (env_name, path))
    path = (os.path.expanduser("~") + "/anaconda/envs/" + env_name + "/bin") + os.pathsep +  path
    logging.info("PATH NOW: %s" % path)
    os.environ["PATH"] = path

    # install dependencies
    # (TODO: handle specific versions of numpy/scipy)
    get_exitcode_stdout_stderr("pip install git+https://github.com/swryan/testflo@work")

    install_cmd = "pip install "
    for dependency in dependencies:
        code, out, err = get_exitcode_stdout_stderr(install_cmd + dependency)

    install_cmd = "python setup.py install"
    for trigger in triggers:
        with repo(trigger, repo_name):
            code, out, err = get_exitcode_stdout_stderr(install_cmd)


def remove_env(env_name, keep_env):
    """
    Deactivate and remove a conda env at the end of a benchmarking run.
    """
    logging.info("PATH AT FIRST: " + os.environ["PATH"])
    path = os.environ["PATH"].split(os.pathsep)
    path.remove(path[0])  # remove modified
    path = (os.pathsep).join(path)
    path = ((os.environ["KEEP_PATH"]) + os.pathsep +  path)
    logging.info("PATH NOW: %s" % path)
    os.environ["PATH"] = path

    if(not keep_env):
        conda_delete = "conda env remove -y --name " + env_name
        code, out, err = get_exitcode_stdout_stderr(conda_delete)
        return code

def remove_repo_dir(repo_dir):
    """
    Remove repo directory before a benchmarking run.
    Will force fresh cloning and avoid branch issues.
    """
    remove_cmd = "rm -rf " + repo_dir

    if os.path.exists(repo_dir):
        code, out, err = get_exitcode_stdout_stderr(remove_cmd)

def run_benchmarks(csv_file):
    """
    Use testflo to run benchmarks)
    """
    testflo_cmd = "testflo -bv -d %s" % csv_file
    code, out, err = get_exitcode_stdout_stderr(testflo_cmd)
    print(code, out, err)


def plot_benchmark_data(project, spec):
    logging.info('plot: %s, %s' % (project, spec))
    try:
        import numpy as np
        from matplotlib import pyplot

        db = BenchmarkDatabase(project)

        c = db.cursor

        data = {}
        for row in c.execute("SELECT * FROM BenchmarkData WHERE Spec=? and Status=='OK' ORDER BY DateTime", (spec,)):
            logging.info('row: %s' % str(row))
            data.setdefault('timestamp', []).append(row[0])
            data.setdefault('status', []).append(row[2])
            data.setdefault('elapsed', []).append(row[3])
            data.setdefault('memory', []).append(row[4])

        timestamp = np.array(data['timestamp'])
        elapsed   = np.array(data['elapsed'])
        maxrss    = np.array(data['memory'])

        fig, a1 = pyplot.subplots()
        x = np.array(range(len(timestamp)))

        a1.plot(x, elapsed, 'b-')
        a1.set_xlabel('run#')
        a1.set_ylabel('elapsed', color='b')
        for tl in a1.get_yticklabels():
            tl.set_color('b')

        a2 = a1.twinx()
        a2.plot(x, maxrss, 'r-')
        a2.set_ylabel('maxrss', color='r')
        for tl in a2.get_yticklabels():
            tl.set_color('r')

        pyplot.title(spec)
        pyplot.show()
    except ImportError:
        raise RuntimeError("numpy and matplotlib are required to plot benchmark data.")


def _get_parser():
    """Returns a parser to handle command line args."""

    parser = ArgumentParser()
    parser.usage = "benchmark [options]"

    parser.add_argument('projects', metavar='project', nargs='*',
                        help='project to benchmark (references a JSON file in the working directory)')

    parser.add_argument('-p', '--plot', metavar='SPEC', action='store', dest='plot',
                        help='the spec of a benchmark to plot')

    parser.add_argument('-f', '--force', action='store_true', dest='force',
                        help='do the benchmark even if nothing has changed')

    parser.add_argument('-k', '--keep-env', action='store_true', dest='keep_env',
                        help='keep the created conda env after execution (usually for troubleshooting purposes)')

    return parser


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # read local configuration if available
    try:
        conf.update(read_json("benchmark.cfg"))
    except IOError:
        pass

    options = _get_parser().parse_args(args)

    with cd(conf["working_dir"]):
        for project in options.projects:

            # get project info
            if project.endswith(".json"):
                project_file = project
            else:
                project_file = project+".json"
            project_info = read_json(project_file)
            project_info["name"] = os.path.basename(project_file).rsplit('.', 1)[0]

            # run benchmark or plot as requested
            if options.plot:
                plot_benchmark_data(project_info["name"], options.plot)
            else:
                benchmark(project_info, force=options.force, keep_env=options.keep_env)


if __name__ == '__main__':
    sys.exit(main())
