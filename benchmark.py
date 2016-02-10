#!/hx/u/kmarstel/anaconda/bin/python

from subprocess import call, Popen, PIPE
import os, shlex, sqlite3, yaml, time, csv
from contextlib import contextmanager

def benchmark(project):
    update_triggered_by = []
    benchmark_needed = 0
    project_dir = "%s%s" % (os.path.expanduser("~/"), project)

    with cd(project_dir):
        #connect to database, prepare to execute cmds with cursor
        dbname = "benchmark.db"
        conn = sqlite3.connect(dbname)
        c = conn.cursor()

        #open yaml config file, parse it.
        with open("benchmark.yaml", 'r') as stream:
            yaml_data = yaml.load(stream)
        
        #check local project for latest commit
        last_benched_commit_project = get_last_benchmarked_commit(c, project) 
        current_commit_project = get_current_commit_project(project, str(yaml_data["branch"]))
        if (last_benched_commit_project != current_commit_project):
            update_triggered_by.append(project)
            benchmark_needed = 1 
         
        for dependency in yaml_data["dependencies"]:
            dep_name = dependency.split('/')[-1]
            print(dep_name)
            last_benched_commit = str(get_last_benchmarked_commit(c, dependency))
            print ("Last Benched Commit: " + last_benched_commit)
            current_commit = str(get_current_commit_deps(dependency))
            print ("Current Commit: " + current_commit)
            if (last_benched_commit != current_commit):
                print("%s is out of date.\n\n" % dep_name)
                update_triggered_by.append(dependency)
                benchmark_needed = 1

        if (benchmark_needed):
            print("Benchmark triggered by updates to: ", update_triggered_by)
            conda_env = create_conda_env(project, yaml_data["dependencies"])
            activate_install_conda_env(conda_env, yaml_data["dependencies"])
            run_testflo_benchmarks()
            read_benchmark_data(c)
            remove_conda_env(conda_env)

        conn.commit()
        conn.close()

@contextmanager
def cd(newdir):
    """
    A cd that will better handle error and return to its orig dir.
    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

#Currently using this to get debug information on calls
def get_exitcode_stdout_stderr(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """
    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    return exitcode, out, err

def get_last_benchmarked_commit(cursor, dependency):
    """
    Check the database for the most recent commit that was benchmarked.
    """
    execute_string = "SELECT LatestCommit FROM DependencyCommits WHERE Repo == "
    execute_string += "'" + dependency + "'"
    cursor.execute(execute_string)
    commit = cursor.fetchall()
    return commit

def get_current_commit_project(project, branch):
    """
    Chcek the local project for it's latest commit.
    """
    #first check the project to see if it has changed.
    git_pull = "git pull origin " + branch
    code, out, err = get_exitcode_stdout_stderr(git_pull)
    code, out, err = get_exitcode_stdout_stderr("git rev-parse HEAD")
    return out

def get_current_commit_deps(dependency):
    """
    Update and check the local dep repo for the most recent commit.
    """
    with cd (os.path.expanduser("~")):
        name = dependency.split('/')[-1]
        repo_dir = "%s%s" % (os.path.expanduser("~/"), name)

        clone_git = "git clone " + dependency
        clone_hg = "hg clone " + dependency

        pull_git = "git pull origin master"
        pull_hg = "hg pull; hg merge"

        commit_git = "git rev-parse HEAD"
        commit_hg = "hg id -i"

        if(not (os.path.isdir(repo_dir))):
            print("Missing:" + repo_dir)
            print("Executing: " + clone_git)
            code, out, err = get_exitcode_stdout_stderr(clone_git)
            print (code, out, err)
            if (code != 0):
                print("Executing: " + clone_git)
            code, out, err = get_exitcode_stdout_stderr(clone_hg)
            print(code, out, err)

        with cd (repo_dir):
            #once in the repo, pull latest commit from desired branch
            code, out, err = get_exitcode_stdout_stderr(pull_git)
            if (code != 0):
                code, out, err = get_exitcode_stdout_stderr(pull_hg)

            #once we have the latest code, get the commit ID
            code, out, err = get_exitcode_stdout_stderr(commit_git)
            if (code != 0):
                code, out, err = get_exitcode_stdout_stderr(commit_hg)
        return out

def create_conda_env(project, dependencies):
    """
    Create a conda env and pip install the latest local reqs into it.
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    env_name = project + "_" + timestr
    conda_create = "conda create -y -n " + env_name + " python=2.7 pip"
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
    print("PATH AT FIRST:"+ os.environ["PATH"])
    path = os.environ["PATH"].split(os.pathsep)
    os.environ["KEEP_PATH"]= path[0] #old path leader
    path.remove(path[0]) #remove default conda env
    path = (os.pathsep).join(path)
    path = (os.path.expanduser("~") + "/anaconda/envs/" + env_name +"/bin") + os.pathsep +  path
    print ("PATH NOW: "+ path)
    os.environ["PATH"] = path

    #install a couple things that must be in there
    get_exitcode_stdout_stderr("pip install numpy")
    get_exitcode_stdout_stderr("pip install scipy")
    get_exitcode_stdout_stderr("pip install mercurial")
    get_exitcode_stdout_stderr("pip install git+https://github.com/naylor-b/testflo.git")

    for dependency in dependencies:
        git_install = "pip install git+" + dependency
        hg_install = "pip install hg+" + dependency

        print("\nInstalling dependency: " + dependency)
        print("Trying git: " + git_install)
        code, out, err = get_exitcode_stdout_stderr(git_install)
        print (code, out, err)
        if (code != 0):
            print("Trying hg: " + hg_install)
            code, out, err = get_exitcode_stdout_stderr(hg_install)
            print (code, out, err)
        print ("\n\n")

    #finally, install the actual project into the conda env
    get_exitcode_stdout_stderr("pip install -e .")

def remove_conda_env(env_name):
    """
    Deactivate and remove a conda env at the end of a benchmarking run.
    """
    print("PATH AT FIRST: "+ os.environ["PATH"])
    path = os.environ["PATH"].split(os.pathsep)
    path.remove(path[0]) #remove modified
    path = (os.pathsep).join(path)
    path = ((os.environ["KEEP_PATH"]) + os.pathsep +  path)
    print ("PATH NOW: " + path)
    os.environ["PATH"] = path

    conda_delete = "conda env remove -y --name " + env_name
    #code, out, err = get_exitcode_stdout_stderr(conda_delete)
    #return code

def run_testflo_benchmarks():
    """
    Run testflo (eventually to run benchmarks)
    """
    testflo_cmd = "testflo "
    code, out, err = get_exitcode_stdout_stderr(testflo_cmd)
    print (code, out, err)

def read_benchmark_data(cursor):
    """
    Stolen directly from Steve, needs to be modified
    """
    benchmarks = {}
    with open('benchmark_data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            spec = row[1].rsplit('.', 1)[1]
            benchmarks.setdefault(spec, {})
            benchmarks[spec].setdefault('timestamp', []).append(row[0])
            benchmarks[spec].setdefault('status', []).append(row[2])
            benchmarks[spec].setdefault('elapsed', []).append(row[3])
            benchmarks[spec].setdefault('maxrss', []).append(row[4])
    from pprint import pprint
    pprint(benchmarks)
    #eventually do something with database cursor to update entries


if __name__ == "__main__":
    #projects = ["OpenMDAO", "pointer", "mission-allocation", "CADRE"]
    projects = ["OpenMDAO"]

    for project in projects:
        benchmark(project)
