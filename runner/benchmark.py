#!/usr/bin/env python
from __future__ import division

import traceback

from subprocess import Popen, PIPE

import sys
import os
import shlex
import sqlite3
import time
import json
import csv
import math

from datetime import datetime

import logging
logging.basicConfig(level=logging.DEBUG)

from argparse import ArgumentParser

from contextlib import contextmanager


# copy current env, we will modify and use for external commands
env = os.environ.copy()


benchmark_dir = os.path.abspath(os.path.dirname(__file__))

#
# default configuration options
#

conf = {
    # directories used during benchmarking
    "working_dir": benchmark_dir,
    "repo_dir":    "repos",
    "logs_dir":    "logs",

    # remove benchmark data file after adding to database
    "remove_csv":  False,

    # generate a plot showing history of each benchmark
    "plot_history": True,

    # CA cert information needed by curl (defaults)
    "ca": {
        "cacert":  "/etc/ssl/certs/ca-certificates.crt",
        "capath":  "/etc/ssl/certs"
    },
}


#
# utility functions
#

def read_json(filename):
    """
    read data from a JSON file
    """
    with open(filename) as json_file:
        return json.loads(json_file.read())


def write_json(filename, data):
    """
    write JSON data to a file
    """
    with open(filename, 'w') as json_file:
        json_file.write(json.dumps(data))


def prepend_path(newdir, path):
    """
    prepend `newdir` to `path`, return modified `path`
    """
    dirs = path.split(os.pathsep)
    dirs.insert(0, newdir)
    path = (os.pathsep).join(dirs)
    return path


def get_exitcode_stdout_stderr(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """
    logging.info("CMD => %s", cmd)
    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE, env=env)
    out, err = proc.communicate()
    rc = proc.returncode
    logging.info("RC => %d", rc)
    if out:
        logging.debug("STDOUT =>\n%s", out)
    if err:
        logging.debug("STDERR =>\n%s", err)
    return rc, out, err


def remove_dir(dirname):
    """
    Remove repo directory before a benchmarking run.
    Will force fresh cloning and avoid branch issues.
    """
    remove_cmd = "rm -rf " + dirname

    if os.path.exists(dirname):
        code, out, err = get_exitcode_stdout_stderr(remove_cmd)


def upload(files, dest):
    """
    upload files to destination via scp
    """
    cmd = "scp %s %s" % (" ".join(files), dest)
    code, out, err = get_exitcode_stdout_stderr(cmd)
    return code


def init_logging():
    """
    initialize logging to stdout with clean format
    """
    log = logging.getLogger()

    # remove old handler(s)
    for hdlr in log.handlers:
        log.removeHandler(hdlr)

    # set stdout handler with clean format
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)


def init_log_file(name):
    """
    initialize logging file with given name
    """
    logs_dir = os.path.expanduser(os.path.join(conf["logs_dir"]))
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    filename = os.path.join(logs_dir, name+".log")
    fh = logging.FileHandler(filename)

    format_str = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    fh.formatter = logging.Formatter(format_str)

    log = logging.getLogger()
    log.addHandler(fh)


#
# context managers
#

@contextmanager
def cd(newdir):
    """
    A cd that will better handle error and return to its orig dir.
    """
    logging.info('cd into %s', newdir)
    prevdir = os.getcwd()
    fulldir = os.path.expanduser(newdir)
    if not os.path.exists(fulldir):
        os.makedirs(fulldir)
    os.chdir(fulldir)
    try:
        yield
    finally:
        logging.info('cd from %s back to %s', fulldir, prevdir)
        os.chdir(prevdir)


@contextmanager
def repo(repository, branch=None):
    """
    cd into local copy of repository.  if the repository has not been
    cloned yet, then clone it to working directory first.
    """
    prev_dir = os.getcwd()

    repo_dir = conf["repo_dir"]
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)
    logging.info('cd into repo dir %s from  %s', repo_dir, prev_dir)
    os.chdir(repo_dir)

    repo_name = repository.split('/')[-1]
    if not os.path.isdir(repo_name):
        clone_repo(repository, branch)

    logging.info('cd into repo %s', repo_name)
    os.chdir(repo_name)
    try:
        yield
    finally:
        logging.info('cd from repo %s back to %s', repo_name, prev_dir)
        os.chdir(prev_dir)


#
# respository helpers
#

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
    git_pull = "git pull"
    git_commit = "git rev-parse HEAD"

    hg_pull = "hg pull"
    hg_merge = "hg merge"
    hg_commit = "hg id -i"

    # pull latest commit from desired branch and get the commit ID
    code, out, err = get_exitcode_stdout_stderr(hg_pull)
    if (code is 0):
        code, out, err = get_exitcode_stdout_stderr(hg_merge)
        code, out, err = get_exitcode_stdout_stderr(hg_commit)
    else:
        code, out, err = get_exitcode_stdout_stderr(git_pull)
        code, out, err = get_exitcode_stdout_stderr(git_commit)

    return out  # .strip()   # TODO: strip the commit IDs in the database as well


#
# anaconda helpers
#

def activate_env(env_name, dependencies, local_repos):
    """
    Create and activate a conda env, install dependencies and then
    any local repositories
    """
    cmd = "conda create -y -q -n " + env_name

    # handle python and numpy/scipy dependencies
    for dep in dependencies:
        if dep.startswith("python") or dep.startswith("numpy") or dep.startswith("scipy"):
            cmd = cmd + " " + dep

    # add other required packages
    conda_pkgs = " ".join([
        "pip",          # for installing dependencies
        "git",          # for cloning git repos
        "mercurial",    # for cloning hg repos
        "swig",         # for building dependencies
        "cython",       # for building dependencies
        "psutil",       # for testflo benchmarking
        "nomkl",        # TODO: experiment with this
        "matplotlib",   # for plotting results
        "curl",         # for uploading files & slack messages
        "sqlite"        # for backing up the database
    ])
    cmd = cmd + " " + conda_pkgs

    code, out, err = get_exitcode_stdout_stderr(cmd)
    if (code != 0):
        raise RuntimeError("Failed to create conda environment", env_name, code, out, err)

    # activate environment by modifying PATH
    path = env["PATH"].split(os.pathsep)
    for dirname in path:
        if "anaconda" in dirname:
            conda_dir = dirname
            path.remove(conda_dir)
            break
    env["ORIG_CONDA_DIR"] = conda_dir
    env["PATH"] = prepend_path(conda_dir.replace("bin",  "envs/"+env_name+"/bin"), (os.pathsep).join(path))

    logging.info("env_name: %s, path: %s", env_name, env["PATH"])

    # need to do a pip install with --prefix to get things installed into proper conda env
    pipinstall = "pip install -q --install-option=\"--prefix=" + conda_dir.replace("bin",  "envs/"+env_name) + "\" "

    # install testflo to do the benchmarking
    code, out, err = get_exitcode_stdout_stderr(pipinstall + "git+https://github.com/OpenMDAO/testflo")
    if (code != 0):
        raise RuntimeError("Failed to install testflo to", env_name, code, out, err)

    # dependencies are pip installed
    for dependency in dependencies:
        # numpy and scipy are installed when the env is created
        if (not dependency.startswith("python=") and not dependency.startswith("numpy") and not dependency.startswith("scipy")):
            code, out, err = get_exitcode_stdout_stderr(pipinstall + os.path.expanduser(dependency))
            if (code != 0):
                raise RuntimeError("Failed to install", dependency, "to", env_name, code, out, err)

    # triggers are installed from a local copy of the repo via 'setup.py install'
    for local_repo in local_repos:
        with repo(local_repo):
            code, out, err = get_exitcode_stdout_stderr("python setup.py install")
            if (code != 0):
                raise RuntimeError("Failed to install", local_repo, "to", env_name, code, out, err)

    return True


def remove_env(env_name, keep_env):
    """
    Deactivate and remove a conda env at the end of a benchmarking run.
    """
    path = env["PATH"].split(os.pathsep)
    logging.info("PATH AT FIRST: %s", env["PATH"])
    for dirname in path:
        if "anaconda" in dirname:
            path.remove(dirname)
            break
    env["PATH"] = prepend_path(env["ORIG_CONDA_DIR"], (os.pathsep).join(path))
    logging.info("PATH NOW: %s", env["PATH"])

    if not keep_env:
        conda_delete = "conda env remove -q -y --name " + env_name
        code, out, err = get_exitcode_stdout_stderr(conda_delete)
        return code


#
# worker classes
#

class Slack(object):
    """
    this class encapsulates the logic required to post messages and files to Slack
    """
    def __init__(self, cfg, ca):
        self.cfg = cfg
        self.ca = ca

    def post_message(self, message):
        """
        post a simple message
        """
        payload = {
            "attachments": [
                {
                    "fallback": message,
                    "pretext":  message,
                    "mrkdwn_in": ["pretext", "fields"]
                }
            ],
            "unfurl_links": "false",
            "unfurl_media": "false"
        }

        return self.post_payload(payload)

    def post_image(self, title, image_url):
        """
        post an image
        """
        payload = {
            "attachments": [
                {
                    "fallback":  title,
                    "pretext":   title,
                    "image_url": image_url
                }
            ],
            "unfurl_links": "true",
            "unfurl_media": "true"
        }

        return self.post_payload(payload)

    def post_payload(self, payload):
        """
        post a payload
        """
        p = json.dumps(payload)
        u = self.cfg["message"]
        c = "--cacert %s --capath %s " % (self.ca["cacert"], self.ca["capath"])

        cmd = "curl -s -X POST -H 'Content-type: application/json' --data '%s' %s %s" % (p, u, c)

        code, out, err = get_exitcode_stdout_stderr(cmd)
        if code:
            logging.warn("Could not post msg to slack: %d\n%s\n%s", code, out, err)

        return code

    def post_file(self, filename, title):
        """
        post a file to slack
        """
        cmd = "curl -s "

        cmd += "-F file=@%s -F title=%s -F filename=%s -F channels=%s -F token=%s " \
             % (filename, title, filename, self.cfg["channel"], self.cfg["token"])

        cmd += "--cacert %s --capath %s  https://slack.com/api/files.upload" \
             % (self.ca["cacert"], self.ca["capath"])

        code, out, err = get_exitcode_stdout_stderr(cmd)

        if code:
            logging.warn("Could not post file to slack:\n%s\n%s", out, err)

        return code


class BenchmarkDatabase(object):
    def __init__(self, name):
        self.name = name
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
        if the benchmark data tables have not been created yet, create them
        """
        # a table containing the status, elapsed time and memory usage for each
        # benchmark in a run
        self.cursor.execute("CREATE TABLE if not exists BenchmarkData"
                            " (DateTime INT, Spec TEXT, Status TEXT, Elapsed REAL, Memory REAL,"
                            "  LoadAvg1m REAL, LoadAvg5m REAL, LoadAvg15m REAL, PRIMARY KEY (DateTime, Spec))")

        # a table containing the versions of all installed dependencies for a run
        self.cursor.execute("CREATE TABLE if not exists InstalledDeps"
                            " (DateTime INT,  InstalledDep TEXT, Version TEXT,"
                            "  PRIMARY KEY (DateTime, InstalledDep))")

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
            logging.info('INSERTING COMMIT %s %s', trigger, commit)
            self.cursor.execute('INSERT OR REPLACE INTO LastCommits VALUES (?, ?)', (trigger, str(commit)))
            self.cursor.execute('INSERT INTO Commits VALUES (?, ?, ?)', (timestamp, trigger, str(commit)))

    def add_benchmark_data(self, commits, filename, installed):
        """
        Insert benchmarks results into BenchmarkData table.
        Create the table if it doesn't already exist.
        """
        self._ensure_benchmark_data()

        data_added = False

        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                logging.info('INSERTING BenchmarkData %s' % str(row))
                try:
                    spec = row[1].rsplit('/', 1)[1]  # remove path from benchmark file name
                    self.cursor.execute("INSERT INTO BenchmarkData VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                                        (row[0], spec, row[2], float(row[3]), float(row[4]),
                                         float(row[5]), float(row[6]), float(row[7])))
                    data_added = True
                except IndexError:
                    logging.warn("Invalid benchmark data found in results:\n %s", str(row))

        if data_added:
            timestamp = row[0]  # row[0] is the timestamp for this set of benchmark data
            self.update_commits(commits, timestamp)
            for dep, ver in installed.items():
                self.cursor.execute("INSERT INTO InstalledDeps VALUES(?, ?, ?)", (timestamp, dep, ver))

    def check_benchmarks(self, timestamp=None):
        """
        Check the benchmark data from the given timestep for any benchmark with
        a 10% or greater change in elapsed time or memory usage.
        If no timestamp is given then check the most recent benchmark data.
        """
        self._ensure_benchmark_data()

        logging.info("Checking benchmarks for timestamp: %s", timestamp)

        if timestamp is None:
            for row in self.cursor.execute("SELECT * FROM BenchmarkData ORDER BY DateTime DESC LIMIT 1"):
                timestamp = row[0]  # row[0] is the timestamp for this set of benchmark data

        if timestamp is None:
            msg = "No benchmark data found"
            logging.warn(msg)
            return []

        date_str = datetime.fromtimestamp(timestamp)

        curr_data = self.get_data_for_timestamp(timestamp)
        if not curr_data:
            msg = "No benchmark data found for timestamp %d (%s)" % (timestamp, date_str)
            logging.warn(msg)
            return []

        prev_time = None
        for row in self.cursor.execute("SELECT * FROM BenchmarkData WHERE DateTime<? and Status=='OK' ORDER BY DateTime DESC LIMIT 1", (timestamp,)):
            prev_time = row[0]  # row[0] is the timestamp for this set of benchmark data

        if not prev_time:
            msg = "No benchmark data found previous to timestamp %d (%s)" % (timestamp, date_str)
            logging.warn(msg)
            return []

        prev_data = self.get_data_for_timestamp(prev_time)

        cpu_messages = []
        mem_messages = []

        for i in range(len(curr_data["spec"])):
            curr_spec    = curr_data["spec"][i]

            curr_elapsed = curr_data["elapsed"][i]
            curr_memory  = curr_data["memory"][i]
            curr_load1   = curr_data["load_1m"][i]
            curr_load5   = curr_data["load_5m"][i]
            curr_load15  = curr_data["load_15m"][i]

            prev_elapsed = prev_data["elapsed"][i]
            prev_memory  = prev_data["memory"][i]
            prev_load1   = prev_data["load_1m"][i]
            prev_load5   = prev_data["load_5m"][i]
            prev_load15  = prev_data["load_15m"][i]

            time_delta   = curr_elapsed - prev_elapsed
            mem_delta    = curr_memory  - prev_memory

            if "url" in conf:
                link = "<%s|%s>" % (conf["url"]+self.name+'/'+curr_spec, curr_spec.split(".")[-1])
            else:
                link = curr_spec.split(".")[-1]

            pct_change = 100.*time_delta/prev_elapsed
            if abs(pct_change) >= 10.:
                inc_or_dec = "decreased" if (pct_change < 0) else "increased"
                msg = "%s %s by %4.1f%%: %5.2f (load avg = %3.1f, %3.1f, %3.1f) vs. %5.2f (load avg = %3.1f, %3.1f, %3.1f)" \
                    % (link, inc_or_dec, abs(pct_change),
                       curr_elapsed, curr_load1, curr_load5, curr_load15,
                       prev_elapsed, prev_load1, prev_load5, prev_load15)
                cpu_messages.append(msg)

            pct_change = 100.*mem_delta/prev_memory
            if abs(pct_change) >= 10.:
                inc_or_dec = "decreased" if (pct_change < 0) else "increased"
                msg = "<%s|%s> %s by %4.1f%% (%5.2f  vs. %5.2f)" \
                    % (link, inc_or_dec, abs(pct_change), curr_memory, prev_memory)
                mem_messages.append(msg)

        return cpu_messages, mem_messages

    def get_data_for_timestamp(self, timestamp):
        """
        get benchmark data for the given timestamp
        """
        data = {}

        for row in self.cursor.execute("SELECT * FROM BenchmarkData WHERE DateTime=? and Status=='OK' ORDER BY DateTime", (timestamp,)):
            data.setdefault('timestamp', []).append(row[0])
            data.setdefault('spec', []).append(row[1])
            data.setdefault('status', []).append(row[2])
            data.setdefault('elapsed', []).append(row[3])
            data.setdefault('memory', []).append(row[4])
            data.setdefault('load_1m', []).append(row[5])
            data.setdefault('load_5m', []).append(row[6])
            data.setdefault('load_15m', []).append(row[7])

        return data

    def get_data_for_spec(self, spec):
        """
        get benchmark data for the given spec
        """
        data = {}

        for row in self.cursor.execute("SELECT * FROM BenchmarkData WHERE Spec=? and Status=='OK' ORDER BY DateTime", (spec,)):
            data.setdefault('timestamp', []).append(row[0])
            data.setdefault('status', []).append(row[2])
            data.setdefault('elapsed', []).append(row[3])
            data.setdefault('memory', []).append(row[4])
            data.setdefault('LoadAvg1m', []).append(row[5])
            data.setdefault('LoadAvg5m', []).append(row[6])
            data.setdefault('LoadAvg15m', []).append(row[7])

        return data

    def dump(self):
        """
        dump database to SQL file
        """
        with open(self.dbname+'.sql', 'w') as f:
            for line in self.conn.iterdump():
                f.write('%s\n' % line)

    def plot_all(self, show=False, save=True):
        """
        generate a history plot of each benchmark
        """
        self._ensure_benchmark_data()

        specs = []
        for row in self.cursor.execute("SELECT DISTINCT Spec FROM BenchmarkData"):
            specs.append(row[0])

        filenames = []
        for spec in specs:
            filenames.append(self.plot_benchmark_data(spec, show=show, save=save))

        return [f for f in filenames if f is not None]

    def get_specs(self):
        specs = []
        for row in self.cursor.execute("SELECT DISTINCT Spec FROM BenchmarkData"):
            specs.append(row[0])
        return specs

    def plot_benchmark_data(self, spec=None, show=False, save=False):
        """
        generate a history plot for a benchmark
        """
        import numpy as np

        logging.info('plot: %s', spec)

        self._ensure_benchmark_data()

        filename = None

        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot, ticker

            data = self.get_data_for_spec(spec)

            if not data:
                logging.warn("No data to plot for %s", spec)
                return

            timestamp = np.array(data['timestamp'])
            elapsed   = np.array(data['elapsed'])
            maxrss    = np.array(data['memory'])

            fig, a1 = pyplot.subplots()
            a1.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
            x = np.array(range(len(timestamp)))

            a1.plot(x, elapsed, 'b-')
            a1.set_xlabel('run#')
            a1.set_ylabel('elapsed', color='b')
            a1.set_ylim(0, max(elapsed)*1.15)
            for tl in a1.get_yticklabels():
                tl.set_color('b')

            a2 = a1.twinx()
            a2.plot(x, maxrss, 'r-')
            a2.set_ylabel('maxrss', color='r')
            a2.set_ylim(0, max(maxrss)*1.15)
            for tl in a2.get_yticklabels():
                tl.set_color('r')

            label = spec.rsplit(':', 1)[1]
            pyplot.title(label.replace(".benchmark_", ": "))

            if show:
                pyplot.show()

            if save:
                filename = spec.replace(":", "_") + ".png"
                pyplot.savefig(filename)
                code, out, err = get_exitcode_stdout_stderr("chmod 644 " + filename)

        except ImportError:
            logging.info("numpy and matplotlib are required to plot benchmark data.")

        return filename

    def plot_benchmarks(self, show=False, save=False):
        """
        generate a history plot for this projects benchmarks
        """
        import numpy as np

        logging.info('plot: %s', self.name)
        filenames = []

        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot, dates

            color_map = pyplot.get_cmap('rainbow')

            mondays = dates.WeekdayLocator(dates.MONDAY)    # major ticks on the mondays
            weekFmt = dates.DateFormatter('%b %d')          # e.g., Jan 12

            self._ensure_benchmark_data()

            # select only the specs that have more than one data point
            specs = self.get_specs()

            select_specs = []
            for spec in specs:
                data = self.get_data_for_spec(spec)
                if data and len(data['elapsed']) > 1:
                    select_specs.append(spec)

            specs = select_specs

            specs_per_plot = 10

            plot_count = int(math.ceil(len(specs)/specs_per_plot))

            for plot_no in range(plot_count):
                pyplot.figure()

                # select up to 'specs_per_plot' specs to plot
                plot_specs = specs[:specs_per_plot]
                specs = specs[specs_per_plot:]

                # initialize max values to normalize data
                max_elapsed = 0
                max_memory = 0

                num_specs = len(plot_specs)
                color_cycle = iter([color_map(1.*i/num_specs) for i in range(num_specs)])

                for spec in plot_specs:
                    data = self.get_data_for_spec(spec)

                    # only plot data for the last num_runs benchmark runs
                    num_runs = 20

                    datetimes = [datetime.fromtimestamp(t) for t in data['timestamp'][-num_runs:]]
                    timestamp = dates.date2num(datetimes)
                    elapsed   = np.array(data['elapsed'][-num_runs:])
                    memory    = np.array(data['memory'][-num_runs:])

                    max_elapsed = max(max_elapsed, np.max(elapsed))
                    max_memory  = max(max_memory, np.max(memory))

                    color = next(color_cycle)

                    a1 = pyplot.subplot(3, 1, 1)
                    pyplot.plot_date(timestamp, elapsed/max_elapsed, '.-', color=color, label=spec)
                    pyplot.ylabel('elapsed time')

                    a2 = pyplot.subplot(3, 1, 2)
                    pyplot.plot_date(timestamp, memory/max_memory, '.-', color=color, label=spec)
                    pyplot.ylabel('memory usage')

                # format the ticks
                a1.set_xticks([])
                a2.xaxis.set_minor_locator(mondays)
                a2.xaxis.set_major_formatter(weekFmt)
                for tick in a2.xaxis.get_major_ticks():
                    tick.label.set_fontsize('x-small')
                pyplot.xticks(rotation=45)

                a1.set_ylim(-0.1, 1.1)
                a2.set_ylim(-0.1, 1.1)

                pyplot.legend(plot_specs, loc=9, prop={'size': 8}, bbox_to_anchor=(0.5, -0.3))

                if show:
                    pyplot.show()

                if save:
                    # unique filename for every hour, recycled every day (24hr cache on Slack?)
                    from time import localtime, strftime
                    filename = self.name + strftime("_%H_", localtime()) + str(plot_no) + ".png"
                    pyplot.savefig(filename)
                    code, out, err = get_exitcode_stdout_stderr("chmod 644 " + filename)
                    filenames.append(filename)

        except ImportError:
            logging.info("numpy and matplotlib are required to plot benchmark data.")

        return filenames

    def backup(self):
        """
        create a local backup database, rsync it to destination
        """
        name = self.dbname
        backup_cmd = "sqlite3 " + name + " \".backup " + name + ".bak\""
        code, out, err = get_exitcode_stdout_stderr(backup_cmd)
        if not code:
            try:
                dest = conf["data"]["upload"]
                rsync_cmd = "rsync -zvh --progress " + name + ".bak " + dest + "/" + name
                code, out, err = get_exitcode_stdout_stderr(rsync_cmd)
            except KeyError:
                pass  # remote backup not configured
            except:
                logging.error("ERROR attempting remote backup")
                logging.error(traceback.format_exc())


class BenchmarkRunner(object):
    """
    this class encapsulates the logic required to conditionally run
    a set of benchmarks if a trigger repository is updated
    """
    def __init__(self):
        if "slack" in conf:
            self.slack = Slack(conf["slack"], conf["ca"])
        else:
            self.slack = None

    def run(self, project, force=False, keep_env=False, unit_tests=False):
        """
        determine if a project or any of it's trigger dependencies have
        changed and run benchmarks if so
        """
        current_commits = {}
        triggered_by = []

        # initialize log file
        timestr = time.strftime("%Y%m%d-%H%M%S")
        run_name = project["name"] + "_" + timestr
        init_log_file(run_name)

        # load the database
        db = BenchmarkDatabase(project["name"])

        # remove any previous repo_dir for this project so we start fresh
        remove_dir(conf["repo_dir"])

        # determine if a new benchmark run is needed, this may be due to the
        # project repo or a trigger repo being updated or the `force` option
        if force:
            triggered_by.append('force')

        triggers = project.get("triggers", [])
        triggers.append(project["repository"])

        for trigger in triggers:
            # for the project repository, we may want a particular branch
            if trigger is project["repository"]:
                branch = project.get("branch", None)
            else:
                branch = None
            # check each trigger for any update since last run
            with repo(trigger, branch):
                msg = 'checking trigger ' + trigger + ' ' + branch if branch else ''
                logging.info(msg)
                current_commits[trigger] = get_current_commit()
                logging.info("Current CommitID: %s", current_commits[trigger])
                last_commit = str(db.get_last_commit(trigger))
                logging.info("Last CommitID: %s", last_commit)
                if (last_commit != current_commits[trigger]):
                    logging.info("There has been an update to %s\n", trigger)
                    triggered_by.append(trigger)

        # if new benchmark run is needed:
        # - create and activate a clean env
        # - run unit tests, if desired
        # - run the benchmark
        # - save benchmark results to database
        # - clean up env and repos
        # - back up database
        if triggered_by:
            logging.info("Benchmark triggered by updates to: %s", str(triggered_by))
            trigger_msg = self.get_trigger_message(project["name"], triggered_by, current_commits)

            triggers = project.get("triggers", [])
            dependencies = project.get("dependencies", [])

            # start out assuming we have a good set of commits
            good_commits = True

            # if unit testing is enabled, then check that we have not already failed unit testing
            if unit_tests:

                # if unit testing fails, the current set of commits will be recorded in fail_file
                fail_file = os.path.join(conf['working_dir'], project["name"]+".fail")

                if os.path.exists(fail_file):
                    good_commits = False
                    failed_commits = read_json(fail_file)
                    for key in current_commits:
                        if current_commits[key] != failed_commits[key]:
                            # there has been a new commit, set flag to r-run and delete fail_file
                            logging.info("found new commit for %s", key)
                            logging.info("old commit: %s", failed_commits[key])
                            logging.info("new commit %s:", current_commits[key])
                            good_commits = True
                            os.remove(fail_file)
                            break

                if not good_commits:
                    logging.info("This set of commits has already failed unit testing.")

            if good_commits or 'force' in triggered_by:

                # activate conda env
                activate_env(run_name, dependencies, triggers)

                with repo(project["repository"], project.get("branch", None)):

                    # install project
                    get_exitcode_stdout_stderr("pip install -q -e .")

                    # run the unit tests if requested and record current_commits if it fails
                    if unit_tests:
                        rc = self.run_unittests(project["name"], trigger_msg)
                        if rc:
                            write_json(fail_file, current_commits)
                            good_commits = False

                    # if we still show good commits, run benchmarks and add data to database
                    if good_commits:

                        # get list of installed dependencies
                        installed_deps = {}
                        rc, out, err = get_exitcode_stdout_stderr("pip list")
                        for line in out.split('\n'):
                            name_ver = line.split(" ", 1)
                            if len(name_ver) == 2:
                                installed_deps[name_ver[0]] = name_ver[1]

                        csv_file = run_name+".csv"
                        self.run_benchmarks(csv_file)
                        db.add_benchmark_data(current_commits, csv_file, installed_deps)

                        # generate plots if requested and upload if image location is provided
                        image_url = None
                        plots = []
                        summary_plots = []
                        images = conf.get("images")
                        if conf["plot_history"]:
                            # plots = db.plot_all()
                            summary_plots = db.plot_benchmarks(save=True)
                            if images and plots + summary_plots:
                                rc = upload(plots + summary_plots, conf["images"]["upload"])
                                if rc == 0:
                                    image_url = conf["images"]["url"]

                        # if slack info is provided, post message to slack and
                        # notify if any benchmarks changed by more than 10%
                        if self.slack:
                            # self.post_results(project["name"], trigger_msg, csv_file, image_url)

                            # post message that benchmarks were run and summary plot(s)
                            self.slack.post_message(trigger_msg)
                            for plot_file in summary_plots:
                                self.slack.post_image("", "/".join([image_url, plot_file]))

                            # check benchmarks for significant changes & post any resulting messages
                            cpu_messages, mem_messages = db.check_benchmarks()
                            notify = "<!channel> The following %s benchmarks had a significant change in %s:\n"

                            # post max_messages at a time
                            max_messages = 9

                            if cpu_messages:
                                self.slack.post_message(notify % (project["name"], "elapsed time"))
                                while cpu_messages:
                                    msg = '\n'.join(cpu_messages[:max_messages])
                                    self.slack.post_message(msg)
                                    cpu_messages = cpu_messages[max_messages:]

                            if mem_messages:
                                self.slack.post_message(notify % (project["name"], "memory usage"))
                                while mem_messages:
                                    msg = '\n'.join(mem_messages[:max_messages])
                                    self.slack.post_message(msg)
                                    mem_messages = mem_messages[max_messages:]

                        if conf["remove_csv"]:
                            os.remove(csv_file)

                        # back up and transfer database
                        db.backup()

                # clean up environment
                remove_env(run_name, keep_env)

    def run_unittests(self, name, trigger_msg):
        testflo_cmd = "testflo -n 1 -vs"

        # run testflo command
        code, out, err = get_exitcode_stdout_stderr(testflo_cmd)
        logging.info(out)
        logging.warn(err)

        # if failure, post to slack
        if code and self.slack:
            self.slack.post_message(trigger_msg + "However, unit tests failed... <!channel>")
            fail_msg = "\"%s : regression testing has failed. See attached results file.\"" % name
            self.slack.post_file("test_report.out", fail_msg)

        return code

    def run_benchmarks(self, csv_file):
        """
        Use testflo to run benchmarks)
        """
        testflo_cmd = "testflo -n 1 -bv -d %s" % csv_file
        code, out, err = get_exitcode_stdout_stderr(testflo_cmd)
        return code

    def get_trigger_message(self, name, triggered_by, current_commits):
        """
        list specific commits (in link form)that caused this bench run
        """
        if "url" in conf:
            pretext = "<%s|%s> benchmarks triggered by " % (conf["url"]+name, name)
        else:
            pretext = "*%s* benchmarks triggered by " % name

        if "force" in triggered_by:
            pretext = pretext + "force:\n"
        else:
            links = []
            # add the specific commit information to each trigger
            for url in triggered_by:
                if "bitbucket" in url:
                    commit = "/commits/"
                else:
                    commit = "/commit/"
                links.append(url + commit + current_commits[url].strip('\n'))

            # insert proper formatting so long URL text is replaced by short trigger-name hyperlink
            links = ["<%s|%s>" % (url.replace("git@github.com:", "https://github.com/"), url.split('/')[-3])
                     for url in links]

            pretext = pretext + "updates to: " + ", ".join(links) + "\n"

        return pretext

    def post_results(self, name, pretext, filename, image_url=None):
        """
        post a message to slack detailing benchmark results
        """
        rows = ""
        names = []
        rslts = []
        color = "good"
        with open(filename, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                try:
                    spec = row[1].rsplit('/', 1)[1]  # remove path from benchmark file name
                    if image_url:
                        image = spec.replace(":", "_") + ".png"
                        plot = "[<%s/%s|History>]" % (image_url, image)
                    else:
                        plot = ""
                    rows = rows + "\t%s \t\tResult: %s \tTime: %5.2f \tMemory: %5.2f \t %s\n" \
                                  % (spec, row[2], float(row[3]), float(row[4]), plot)

                    names.append("```%s```" % spec)
                    rslts.append("```%s\t%8.2fs\t%8.2fmb\t%s```" % (row[2], float(row[3]), float(row[4]), plot))
                    if row[2] != "OK":
                        color = "danger"
                except IndexError:
                    logging.info("Invalid benchmark specification found in results:\n %s", str(row))

            msg_count = int(math.ceil(len(names)/10.))
            for m in range(msg_count):
                if m > 0:
                    pretext = "*%s* benchmarks continued:" % name
                payload = {
                    "attachments": [
                        {
                            "fallback": pretext + rows[:10],
                            "color":    color,
                            "pretext":  pretext,
                            "fields": [
                                { "title": "Benchmark", "value": "\n".join(names[:10]), "short": "true"},
                                { "title": "Results",   "value": "\n".join(rslts[:10]), "short": "true"},
                            ],
                            "mrkdwn_in": ["pretext", "fields"]
                        }
                    ],
                    "unfurl_links": "false",
                    "unfurl_media": "false"
                }

                self.slack.post_payload(payload)

                rows = rows[10:]
                names = names[10:]
                rslts = rslts[10:]


#
# command line
#

def _get_parser():
    """
    Returns a parser to handle command line args.
    """

    parser = ArgumentParser()
    parser.usage = "benchmark [options]"

    parser.add_argument('projects', metavar='project', nargs='*',
                        help='project to benchmark (references a JSON file in the working directory)')

    parser.add_argument('-f', '--force', action='store_true', dest='force',
                        help='do the benchmark even if nothing has changed')

    parser.add_argument('-u', '--unit-tests', action='store_true', dest='unit_tests',
                        help='run the unit tests before running the benchmarks.')

    parser.add_argument('-k', '--keep-env', action='store_true', dest='keep_env',
                        help='keep the created conda env after execution (usually for troubleshooting purposes)')

    parser.add_argument('-p', '--plot', metavar='SPEC', action='store', dest='plot',
                        help='plot benchmark history for SPEC')

    parser.add_argument('-c', '--check', action='store_true', dest='check',
                        help='check the most recent benchmark data for >10%% change')

    parser.add_argument('-d', '--dump', action='store_true', dest='dump',
                        help='dump the contents of the database to an SQL file')

    return parser


def main(args=None):
    """
    process command line arguments and perform requested task
    """
    if args is None:
        args = sys.argv[1:]

    options = _get_parser().parse_args(args)

    # read local configuration if available
    try:
        conf.update(read_json("benchmark.cfg"))
    except IOError:
        pass

    # add any env vars from the config to the working env
    if "env" in conf:
        for key, val in conf["env"].iteritems():
            env[key] = val

    # prepend benchmark dir to PATH to intercept mpirun command
    env["PATH"] = prepend_path(benchmark_dir, env["PATH"])

    # initalize logging to stdout
    init_logging()

    # perform the requested operation for each project
    with cd(conf["working_dir"]):
        for project in options.projects:

            # get project info
            if project.endswith(".json"):
                project_file = project
            else:
                project_file = project+".json"
            project_info = read_json(project_file)
            project_name = os.path.basename(project_file).rsplit('.', 1)[0]
            project_info["name"] = project_name

            # perform requested action, or just run benchmarks
            if options.plot:
                db = BenchmarkDatabase(project_name)
                if options.plot == 'all':
                    db.plot_benchmarks(show=True)
                else:
                    db.plot_benchmark_data(options.plot)
            elif options.dump:
                db = BenchmarkDatabase(project_name)
                db.dump()
            elif options.check:
                db = BenchmarkDatabase(project_name)
                messages = db.check_benchmarks()
                for msg in messages:
                    logging.info(msg)
            else:
                # use a different repo directory for each project
                conf["repo_dir"] = os.path.expanduser(
                    os.path.join(conf["working_dir"], (project_name+"_repos")))

                bm = BenchmarkRunner()
                bm.run(project_info, options.force, options.keep_env, options.unit_tests)


if __name__ == '__main__':
    sys.exit(main())
