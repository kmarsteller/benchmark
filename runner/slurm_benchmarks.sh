#!/bin/bash
#
# This script submits a job via SLURM to perform benchmarks via testlo
#
# Usage: $0 RUN_NAME CSV_FILE NSLOTS
#
#     RUN_NAME : the name of the job (REQUIRED)
#     CSV_FILE : the file name for the benchmark data (REQUIRED)
#     NSLOTS   : the number of processors (optional, default 20)
#

RUN_NAME=$1
CSV_FILE=$2

case $3 in
    ''|*[!0-9]*) NSLOTS=20 ;;
    *)           NSLOTS=$1 ;;
esac

# generate job script
cat << EOM >job
#!/bin/bash 

# USE_PROC_FILES causes I/O erros when using MPI.Spawn
unset USE_PROC_FILES

# create machinefile
srun -l -p mdao /bin/hostname | sort -n | awk '{print \$2}' > slurm.hosts
echo "slurm.hosts:"
cat slurm.hosts

# mpirun
mpirun -np 1 -machinefile slurm.hosts testflo -n 1 -bv -d $CSV_FILE

# release the allocation
exit
EOM

# allocate resources and run the job script (exclude interactive node mdao10)
#salloc -v -v -p mdao -x mdao10 --exclusive --wait-all-nodes=1 -n $NSLOTS -J $RUN_NAME bash job
salloc -v -p mdao -n $NSLOTS -J $RUN_NAME bash job

