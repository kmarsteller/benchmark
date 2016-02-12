#!python
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

# default configuration options
conf = {
    "working_dir": ".",
    "repo_dir": "."
}


class BenchmarkDatabase(object):
    def __init__(self, name):
        self.dbname = name+".db"
        self.conn   = sqlite3.connect(self.dbname)
        self.cursor = self.conn.cursor()

    def get_last_commit(self, dependency):
        """
        Check the database for the most recent commit that was benchmarked.
        """
        c = self.cursor

        # if the commits table has not been created yet, create it
        c.execute("CREATE TABLE if not exists Commits (Dependency TEXT, Repo TEXT, LastCommit TEXT)")

        # try to
        execute_string = "SELECT LastCommit FROM Commits WHERE Repo == "
        execute_string += "'" + dependency + "'"
        c.execute(execute_string)
        commit = c.fetchall()
        print("last commit for", dependency, "=", commit)
        return commit

    def add_benchmark_data(self, filename):
        """
        Insert benchmarks results into BenchmarkData table.
        Create the table if it doesn't already exist.
        """
        c = self.cursor
        c.execute("CREATE TABLE if not exists BenchmarkData (DateTime INT, Spec TEXT, Status TEXT, Elapsed REAL, Memory REAL)")
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                spec = row[1].rsplit('.', 1)[1]
                c.execute("INSERT INTO BenchmarkData VALUES(?, ?, ?, ?, ?)", (row[0], spec, row[2], row[3], row[4]))

    def dump_benchmark_data(self):
        with open(self.dbname+'.benchmark_data.sql', 'w') as f:
            for line in self.conn.iterdump():
                f.write('%s\n' % line)


def read_json(filename):
    """
    read data from a JSON file
    """
    try:
        with open(filename) as json_file:
            return json.loads(json_file.read())
    except:
        raise RuntimeError("%s is not a valid JSON file." % filename)


def get_exitcode_stdout_stderr(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """
    #Currently using this to get debug information on calls
    print('CMD:', cmd)
    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    return exitcode, out, err


@contextmanager
def cd(newdir):
    """
    A cd that will better handle error and return to its orig dir.
    """
    print('cd into', newdir)
    prevdir = os.getcwd()
    fulldir = os.path.expanduser(newdir)
    if not os.path.exists(fulldir):
        os.makedirs(fulldir)
    os.chdir(fulldir)
    try:
        yield
    finally:
        print('cd from', fulldir, 'back to', prevdir)
        os.chdir(prevdir)


@contextmanager
def repo(repository):
    """
    cd into local copy of repository.  if the repository has not been
    cloned yet, then clone it to working directory first.
    """
    prev_dir = os.getcwd()

    repo_dir = os.path.expanduser(conf["repo_dir"])
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)
    print('cd into repo dir', repo_dir, 'from', prev_dir)
    os.chdir(repo_dir)

    repo_name = repository.split('/')[-1]
    if not os.path.isdir(repo_name):
        clone_repo(repository)

    print('cd into repo', repo_name)
    os.chdir(repo_name)
    try:
        yield
    finally:
        print('cd from repo', repo_name, 'back to', prev_dir)
        os.chdir(prev_dir)


def benchmark(project):
    project_info = read_json(project+".json")
    project_info["name"] = project

    update_triggered_by = []
    benchmark_needed = False

    db = BenchmarkDatabase(project)

    dependencies = project_info["dependencies"]
    dependencies.append(project_info["repository"])

    for dependency in dependencies:
        with repo(dependency):
            last_commit = str(db.get_last_commit(dependency))
            print ("Last Commit: " + last_commit)
            current_commit = get_current_commit()
            print ("Current Commit: " + current_commit)
            if (last_commit != current_commit):
                print("There has been an update to %s.\n\n" % dependency)
                update_triggered_by.append(dependency)
                benchmark_needed = 1

    if (benchmark_needed):
        print("Benchmark triggered by updates to: ", update_triggered_by)
        conda_env = create_conda_env(project)
        activate_install_conda_env(conda_env, project_info["dependencies"])
        with repo(project_info["repository"]):
            get_exitcode_stdout_stderr("pip install -e .")
            run_benchmarks()
            db.add_benchmark_data("benchmark_data.csv")

        db.dump_benchmark_data()
        remove_conda_env(conda_env)


def clone_repo(repository):
    """
    clone repository into current directory
    """
    clone_cmd = " clone " + repository

    code, out, err = get_exitcode_stdout_stderr("git" + clone_cmd)
    if (code != 0):
        code, out, err = get_exitcode_stdout_stderr("hg" + clone_cmd)

    print(code, out, err)

    if code:
        raise RuntimeError("Could not clone %s" % repository)


def get_current_commit():
    """
    Update and check the current repo for the most recent commit.
    """
    pull_git = "git pull origin master"
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


def create_conda_env(project):
    """
    Create a conda env.
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    env_name = project + "_" + timestr
    conda_create = "conda create -y -n " + env_name + " python=2.7 pip numpy scipy swig"
    code, out, err = get_exitcode_stdout_stderr(conda_create)
    if (code == 0):
        return env_name
    else:
        print("Falied to create conda environment", env_name)


def activate_install_conda_env(env_name, dependencies):
    """
    Activate an existing conda env; install dependencies into it
    """
    #screwing with path instead of activating.
    print("PATH AT FIRST:" + os.environ["PATH"])
    path = os.environ["PATH"].split(os.pathsep)
    os.environ["KEEP_PATH"] = path[0]  # old path leader
    path.remove(path[0])  # remove default conda env
    path = (os.pathsep).join(path)
    path = (os.path.expanduser("~") + "/anaconda/envs/" + env_name + "/bin") + os.pathsep +  path
    print ("PATH NOW: " + path)
    os.environ["PATH"] = path

    #install a couple things that must be in there
    # get_exitcode_stdout_stderr("conda install numpy")
    # get_exitcode_stdout_stderr("conda install scipy")
    # get_exitcode_stdout_stderr("pip install mercurial")
    get_exitcode_stdout_stderr("pip install git+https://github.com/swryan/testflo@work")

    install_cmd = "python setup.py install"
    for dependency in dependencies:
        with repo(dependency):
            code, out, err = get_exitcode_stdout_stderr(install_cmd)
            print (code, out, err)


def remove_conda_env(env_name):
    """
    Deactivate and remove a conda env at the end of a benchmarking run.
    """
    print("PATH AT FIRST: " + os.environ["PATH"])
    path = os.environ["PATH"].split(os.pathsep)
    path.remove(path[0])  # remove modified
    path = (os.pathsep).join(path)
    path = ((os.environ["KEEP_PATH"]) + os.pathsep +  path)
    print ("PATH NOW: " + path)
    os.environ["PATH"] = path

    conda_delete = "conda env remove -y --name " + env_name
    code, out, err = get_exitcode_stdout_stderr(conda_delete)
    return code


def run_benchmarks():
    """
    Use testflo to run benchmarks)
    """
    testflo_cmd = "testflo -bv"
    code, out, err = get_exitcode_stdout_stderr(testflo_cmd)
    print (code, out, err)


def plot_benchmark_data(project, spec):
    print('plot:', project, spec)
    try:
        import numpy as np
        from matplotlib import pyplot

        db = BenchmarkDatabase(project)

        c = db.cursor

        data = {}
        for row in c.execute("SELECT * FROM BenchmarkData WHERE Spec=? and Status=='OK' ORDER BY DateTime", (spec,)):
            print('row:', row)
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
        print("numpy and matplotlib are required to plot benchmark data.")

def _get_parser():
    """Returns a parser to handle command line args."""

    parser = ArgumentParser()
    parser.usage = "benchmark [options]"

    parser.add_argument('projects', metavar='project', nargs='*',
                        help='project to benchmark (references a JSON file in the working directory')

    parser.add_argument('-p', '--plot', metavar='SPEC', action='store', dest='plot',
                        help='the spec of a benchmark to plot.')

    return parser


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    conf.update(read_json("benchmark.cfg"))

    options = _get_parser().parse_args(args)

    if options.plot:
        with cd(conf["working_dir"]):
            for project in options.projects:
                plot_benchmark_data(project, options.plot)
    else:
        with cd(conf["working_dir"]):
            for project in options.projects:
                benchmark(project)


if __name__ == '__main__':
    sys.exit(main())
