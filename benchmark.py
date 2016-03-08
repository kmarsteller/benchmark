#!/usr/bin/env python
from __future__ import print_function

import traceback

from subprocess import Popen, PIPE

import sys
import os
import shlex
import sqlite3
import time
import json
import csv

import logging
logging.basicConfig(level=logging.DEBUG)

from argparse import ArgumentParser

from contextlib import contextmanager

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
        if the benchmark data tables have not been created yet, create them
        """
        # a table containing the status, elapsed time and memory usage for each
        # benchmark in a run
        self.cursor.execute("CREATE TABLE if not exists BenchmarkData"
                            " (DateTime INT, Spec TEXT, Status TEXT, Elapsed REAL, Memory REAL,"
                            "  PRIMARY KEY (DateTime, Spec))")

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
            logging.info('INSERTING COMMIT %s %s' % (trigger, commit))
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
                    self.cursor.execute("INSERT INTO BenchmarkData VALUES(?, ?, ?, ?, ?)", (row[0], spec, row[2], float(row[3]), float(row[4])))
                    data_added = True
                except IndexError:
                    print("Invalid benchmark specification found in results:\n %s" % str(row))

        if data_added:
            timestamp = row[0]  # row[0] is the timestamp for this set of benchmark data
            self.update_commits(commits, timestamp)
            for dep, ver in installed.items():
                self.cursor.execute("INSERT INTO InstalledDeps VALUES(?, ?, ?)", (timestamp, dep, ver))

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

    def plot_benchmark_data(self, spec=None, show=True, save=False):
        """
        generate a history plot for a benchmark
        """
        logging.info('plot: %s' % spec)
        print('plot: %s' % spec)

        self._ensure_benchmark_data()

        filename = None

        try:
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot, ticker

            data = {}
            for row in self.cursor.execute("SELECT * FROM BenchmarkData WHERE Spec=? and Status=='OK' ORDER BY DateTime", (spec,)):
                logging.info('row: %s' % str(row))
                data.setdefault('timestamp', []).append(row[0])
                data.setdefault('status', []).append(row[2])
                data.setdefault('elapsed', []).append(row[3])
                data.setdefault('memory', []).append(row[4])

            if not data:
                logging.warn("No data to plot for %s" % spec)
                print("No data to plot for %s" % spec)
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
            print("numpy and matplotlib are required to plot benchmark data.")
            print(traceback.format_exc())

        return filename


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
        print(err)
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
def repo(repository, branch=None):
    """
    cd into local copy of repository.  if the repository has not been
    cloned yet, then clone it to working directory first.
    """
    prev_dir = os.getcwd()

    repo_dir = conf["repo_dir"]
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)
    logging.info('cd into repo dir %s from  %s' % (repo_dir, prev_dir))
    os.chdir(repo_dir)

    repo_name = repository.split('/')[-1]
    if not os.path.isdir(repo_name):
        clone_repo(repository, branch)

    logging.info('cd into repo %s' % repo_name)
    print('cd into repo %s' % repo_name)
    os.chdir(repo_name)
    try:
        yield
    finally:
        logging.info('cd from repo %s back to %s' % (repo_name, prev_dir))
        os.chdir(prev_dir)


def benchmark(project_info, force=False, keep_env=False):
    """
    determine if a project or any of it's trigger dependencies have
    changed and run benchmarks if so
    """
    current_commits = {}
    update_triggered_by = []

    timestr = time.strftime("%Y%m%d-%H%M%S")
    run_name = project_info["name"] + "_" + timestr
    init_log(run_name)

    db = BenchmarkDatabase(project_info["name"])

    # remove any previous repo_dir for this project so we start fresh
    remove_repo_dir(conf["repo_dir"])

    # determine if a new benchmark run is needed, this may be due to the
    # project repo or a trigger repo being updated or the `force` option
    if force:
        update_triggered_by.append('force')
    else:
        triggers = project_info.get("triggers", [])
        triggers.append(project_info["repository"])

        for trigger in triggers:
            # for the project repository, we may want a particular branch
            if trigger is project_info["repository"]:
                branch = project_info.get("branch", None)
            else:
                branch = None
            # check each trigger for any update since last run
            with repo(trigger, branch):
                print('checking trigger', trigger, branch if branch else '')
                last_commit = str(db.get_last_commit(trigger))
                logging.info("Last CommitID: %s" % last_commit)
                current_commits[trigger] = get_current_commit()
                logging.info("Current CommitID: %s" % current_commits[trigger])
                if (last_commit != current_commits[trigger]):
                    logging.info("There has been an update to %s\n" % trigger)
                    print("There has been an update to %s" % trigger)
                    update_triggered_by.append(trigger)

    # if new benchmark run is needed:
    # - create and activate a clean env
    # - run the benchmark
    # - save benchmark results to database
    # - clean up env and repos
    if update_triggered_by:
        logging.info("Benchmark triggered by updates to: %s" % str(update_triggered_by))
        print("Benchmark triggered by updates to: %s" % str(update_triggered_by))

        triggers = project_info.get("triggers", [])
        dependencies = project_info.get("dependencies", [])

        create_env(run_name, dependencies)
        activate_env(run_name, triggers, dependencies)

        with repo(project_info["repository"], project_info.get("branch", None)):
            # install project
            get_exitcode_stdout_stderr("pip install -e .")

            # get list of installed dependencies
            installed_deps = {}
            rc, out, err = get_exitcode_stdout_stderr("pip list")
            for line in out.split('\n'):
                name_ver = line.split(" ", 1)
                if len(name_ver) == 2:
                    installed_deps[name_ver[0]] = name_ver[1]

            # run benchmarks and add data to database
            csv_file = run_name+".csv"
            run_benchmarks(csv_file)
            db.add_benchmark_data(current_commits, csv_file, installed_deps)

            # generate plots if requested and upload if image location is provided
            images = conf.get("images")
            if conf["plot_history"]:
                plots = db.plot_all()
                if images and plots:
                    upload(plots, conf["images"]["upload"])

            # if slack info is provided, post message to slack
            if "slack" in conf:
                if images:
                    post_message_to_slack(project_info["name"], update_triggered_by, csv_file, plots)
                else:
                    post_message_to_slack(project_info["name"], update_triggered_by, csv_file)

            if conf["remove_csv"]:
                os.remove(csv_file)

        # clean up environment
        remove_env(run_name, keep_env)

        #back up and transfer database
        backup_db(project_info["name"] + ".db")


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

    return out


def create_env(env_name, dependencies):
    """
    Create a conda env
    """
    cmd = "conda create -y -n " + env_name

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


def activate_env(env_name, triggers, dependencies):
    """
    Activate an existing conda env and install triggers and dependencies into it
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

    # workaround for current issue with anaconda libgfortran
    os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libgfortran.so.3"

    # install testflo to do the benchmarking
    code, out, err = get_exitcode_stdout_stderr("pip install git+https://github.com/naylor-b/testflo")
    if (code != 0):
        raise RuntimeError("Failed to install testflo to", env_name, code, out, err)

    # dependencies are pip installed
    for dependency in dependencies:
        # numpy and scipy are installed when the env is created
        if (not dependency.startswith("python=") and not dependency.startswith("numpy") and not dependency.startswith("scipy")):
            code, out, err = get_exitcode_stdout_stderr("pip install " + os.path.expanduser(dependency))
            if (code != 0):
                raise RuntimeError("Failed to install", dependency, "to", env_name, code, out, err)

    # triggers are installed from a local copy of the repo via 'setup.py install'
    for trigger in triggers:
        with repo(trigger):
            code, out, err = get_exitcode_stdout_stderr("python setup.py install")
            if (code != 0):
                raise RuntimeError("Failed to install", trigger, "to", env_name, code, out, err)

    return True


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


def upload(files, dest):
    """
    upload files to destination via scp
    """
    cmd = "scp %s %s" % (" ".join(files), dest)
    code, out, err = get_exitcode_stdout_stderr(cmd)


def backup_db(name):
    """
    create a local backup database, rsync it to destination
    """
    backup_cmd = "sqlite3 " + name + " \".backup " + name + ".bak\""
    code, out, err = get_exitcode_stdout_stderr(backup_cmd)
    if not code:
        try:
            dest = conf["data"]["upload"]
            rsync_cmd = "rsync -zvh --progress " + name + ".bak " + dest + "/" + name
            code, out, err = get_exitcode_stdout_stderr(rsync_cmd)
        except KeyError:
            pass # remote backup not configured
        except:
            print("ERROR attempting remote backup")


def post_message_to_slack(name, update_triggered_by, filename, plots=None):
    """
    post a message to slack detailing benchmark results
    """
    pretext = "*%s* benchmark tiggered by " % name
    if len(update_triggered_by) == 1 and "force" in update_triggered_by:
        pretext = pretext + "force:\n"
    else:
        links = ["<%s>" % url.replace("git@github.com:", "https://github.com/")
                 for url in update_triggered_by]
        pretext = pretext + "updates to: " + ", ".join(links) + "\n"

    if plots:
        image_url = conf["images"]["url"]

    rows = ""
    name = []
    rslt = []
    color = "good"
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                spec = row[1].rsplit('/', 1)[1]  # remove path from benchmark file name
                if plots:
                    image = spec.replace(":", "_") + ".png"
                    plot = "[<%s/%s|History>]" % (image_url, image)
                else:
                    plot = ""
                rows = rows + "\t%s \t\tResult: %s \tTime: %5.2f \tMemory: %5.2f \t %s\n" \
                              % (spec, row[2], float(row[3]), float(row[4]), plot)

                name.append("```%s```" % spec)
                rslt.append("```%s\t%8.2fs\t%8.2fmb\t%s```" % (row[2], float(row[3]), float(row[4]), plot))
                if row[2] != "OK":
                    color = "danger"
            except IndexError:
                print("Invalid benchmark specification found in results:\n %s" % str(row))

    payload = {
        "attachments": [
            {
                "fallback": pretext + rows,
                "color":    color,
                "pretext":  pretext,
                "fields": [
                    { "title": "Benchmark", "value": "\n".join(name), "short": "true"},
                    { "title": "Results",   "value": "\n".join(rslt), "short": "true"},
                ],
                "mrkdwn_in": ["pretext", "fields"]
            }
        ],
        "unfurl_links": "false",
        "unfurl_media": "false"
    }

    url = conf["slack"]["message"]
    msg = json.dumps(payload)

    if "ca" in conf:
        cacert  = conf["ca"]["cacert"]
        capath  = conf["ca"]["capath"]
        ca = "--cacert %s --capath %s "  % (cacert, capath)
    else:
        ca = ""

    cmd = "curl -X POST -H 'Content-type: application/json' --data '%s' %s %s"  % (msg, url, ca)

    code, out, err = get_exitcode_stdout_stderr(cmd)
    if code:
        logging.warn("Could not post msg to slack", code, out, err)
        print("Could not post msg to slack", code, out, err)


def post_file_to_slack(filename):
    """
    post a file to slack
    """
    channel = conf["slack"]["channel"]
    token   = conf["slack"]["token"]

    cacert  = conf["ca"]["cacert"]
    capath  = conf["ca"]["capath"]

    cmd_fmt = "curl -F file=@%s -F filename=%s -F channels=%s -F token=%s " \
            + "--cacert %s --capath %s https://slack.com/api/files.upload"

    cmd = cmd_fmt % (filename, filename, channel, token, cacert, capath)
    code, out, err = get_exitcode_stdout_stderr(cmd)

    if code:
        logging.warn("Could not post file to slack", code, out, err)
        print("Could not post file to slack", code, out, err)


def init_log(name):
    """
    initialize logging file with given name
    """
    log = logging.getLogger()

    # remove old handler(s)
    for hdlr in log.handlers:
        log.removeHandler(hdlr)

    # set the new handler
    logs_dir = os.path.expanduser(os.path.join(conf["logs_dir"]))
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    filename = os.path.join(logs_dir, name+".log")
    fh = logging.FileHandler(filename)

    format_str = '%(asctime)s %(name)s %(levelname)s: %(message)s'
    fh.formatter = logging.Formatter(format_str)

    log.addHandler(fh)


def _get_parser():
    """
    Returns a parser to handle command line args.
    """

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

    parser.add_argument('-d', '--dump', action='store_true', dest='dump',
                        help='dump the contents of the database to an SQL file')

    return parser


def main(args=None):
    """
    process command line arguments and perform requested task
    """
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
            project_name = os.path.basename(project_file).rsplit('.', 1)[0]
            project_info["name"] = project_name

            # use a different repo directory for each project
            conf["repo_dir"] = os.path.expanduser(
                os.path.join(conf["working_dir"], (project_name+"_repos")))

            # run benchmark or plot as requested
            if options.plot:
                db = BenchmarkDatabase(project_name)
                if options.plot == 'all':
                    db.plot_all()
                else:
                    db.plot_benchmark_data(options.plot)
            elif options.dump:
                db = BenchmarkDatabase(project_name)
                db.dump()
            else:
                benchmark(project_info, force=options.force, keep_env=options.keep_env)


if __name__ == '__main__':
    sys.exit(main())
