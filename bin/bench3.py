#!/usr/bin/env python3
import os
import sys
import glob
import subprocess
import signal
import datetime
import time
import concurrent.futures
import multiprocessing
from multiprocessing.pool import ThreadPool
import re
import numpy as np

from collections import namedtuple
from operator import attrgetter

# arguments
TIMEOUT     = 5200.0
PROBLEMS_SET = "instances/**"
PROBLEMS = PROBLEMS_SET + "/*.cnf*"
RESULTS_DIR = "results"

# data
CSV_HEADER  = "Instance,Result,Time, conflict_size, average_len, reduction, i_uip_reduction, mem_use, core_clause, lbd, attempt_rate, success_rate \n"
Result      = namedtuple('Result', ('problem', 'result', 'elapsed', 'conflict_size', 'average_len', 'reduction', 'i_uip_reduction', 'mem_use', 'core_clause', 'lbd', 'attempt_rate', 'success_rate'))

# constants
SAT_RESULT     = 'sat'
UNSAT_RESULT   = 'unsat'
UNKNOWN_RESULT = 'unknown'
TIMEOUT_RESULT = 'timeout (%.1f s)' % TIMEOUT
ERROR_RESULT   = 'error'

SOLVERS = {
    #timeout is a little more than TIMEOUT
    #"i-uip-realistic"   : "./maplesat_static -i-uip -smart-learn -cpu-lim=1200",
    #"i-uip-reg"  : "./maplesat_static -i-uip -cpu-lim=300",
    #"i-uip-static-2"  : "./maplesat_static -i-uip -static-target  -i-uip-init=2 -cpu-lim=300",
    #"i-uip-cost-eff-2"  : "./maplesat_static -i-uip -cost-eff  -i-uip-init=2 -cpu-lim=300",
    #"i-uip-activity"  : "./maplesat_static -i-uip -activity-uip  -cpu-lim=200",
    #"i-uip-lbd-smart"  : "./maplesat_static -i-uip -lbd-uip -smart-learn  -cpu-lim=2000",
    #"i-uip-mini"  : "./minisat_static -i-uip -i-mini -cpu-lim=2000",
    #i-uip"  : "./minisat_static -i-uip -cpu-lim=2000",
    #"i-uip-mini-active-short"  : "./minisat_static_shortcut -i-uip -i-mini -i-active  -cpu-lim=4000 ",
    #"i-uip-mini-active_new"  : "./minisat_static -i-uip -i-mini -i-active -cpu-lim=5000 ",
    #"i-uip-mini-greedy-dual_short"  : "./minisat_static_shortcut -i-uip -i-mini -i-active-greedy  -i-dual -cpu-lim=4000 ",

    #"i-uip-mini-greedy_new"  : "./minisat_static -i-uip -i-mini -i-active-greedy  -cpu-lim=5000 ",
    "i-uip-mini-active-greedy_new"  : "./minisat_static -i-uip -i-mini -i-active -i-active-greedy  -cpu-lim=5000 "
    #"i-uip-mini-greedy-dual-visid_new"  : "./minisat_static -i-uip -i-mini -i-active-greedy -i-dual -i-visid -cpu-lim=5000 ",
    #"i-uip-mini-active greedy-dual_new"  : "./minisat_static -i-uip -i-mini -i-active -i-active-greedy -i-dual  -cpu-lim=5000 ",
    #"i-uip-mini"  : "./minisat_static -i-uip -i-mini  -cpu-lim=2000 ",
    #"i-uip-mini-active"  : "./minisat_static -i-uip -i-mini -i-active  -cpu-lim=2000 ",

    #"i-uip-mini-new"  : "./minisat_static -i-uip -i-mini -cpu-lim=5000",
    #"1-uip_neo": "./maplesat_static_mult  -cpu-lim=300",
    #"i-uip-lbd-mult": "./maplesat_static_mult  -i-uip -lbd-uip -cpu-lim=5000",
    #"1-uip": "./minisat_static -cpu-lim=5000"
}

def output2result(problem, output):
    # it's important to check for unsat first, since sat
    # is a substring of unsat
    if 'UNSATISFIABLE' in output or 'unsat' in output:
        return UNSAT_RESULT
    if 'SATISFIABLE' in output or 'sat' in output:
        return SAT_RESULT
    if 'UNKNOWN' in output or 'unknown' in output:
        return UNKNOWN_RESULT
    if 'INDETERMINATE' in output:
        return TIMEOUT_RESULT

    # print(problem, ': Couldn\'t parse output', file=sys.stderr)
    return ERROR_RESULT

def get_avg_LBD(output):
    avg_LBD_literals = re.search("Avg LBD\s+:\s+(\d+\.\d+)\s+", output)
    if avg_LBD_literals is not None:
        return avg_LBD_literals.group(1)
    else:
        return "NAN"


def get_reward_score(output):
    reward_literals = re.search("Conflicts per Decision\s+:\s+(\d+\.\d+)\s+", output)
    if reward_literals is not None:
        return reward_literals.group(1)
    else:
        return "NAN"

def get_lbd(output):
    lbd = re.search("lbds\s+:\s*(\d+\.\d+)\s+", output)
    if lbd is not None:
        lbd = lbd.group(1)
    else:
        lbd = 0
    return lbd

def get_attempt_rate(output):
    attempts = re.search("i-uip-attempt-percentage\s+:\s*(0)\s+", output)
    if attempts is not None:
        return 0 , 0
    else:
        attempts  = re.search("i-uip-attempt-percentage\s+:\s*(\d+\.\d+)\s+", output)
        if attempts is not None:
            attempts = attempts.group(1)
            success = re.search("i-uip-percentage\s+:\s*(\d+\.\d+)\s+", output)
            if success is not None:
                return attempts, success.group(1)
            else:
                return attempts, 0

    return 0 , 0


def get_core_clause(output):
    core_clauses = re.search("core clauses\s+:\s+(\d+)\s+", output)
    if core_clauses is not None:
        core_clauses = core_clauses.group(1)
    else:
        core_clauses = 0
    return core_clauses

def get_average_conflict_cluase_size(output):
    conflict_literals = re.search("conflict literals\s+:\s+(\d+)\s+", output)
    conflicts = re.search("conflicts\s+:\s+(\d+)\s+", output)
    conflicts_num = int(conflicts.group(1))
    if conflict_literals is not None:
        average_len = int(conflict_literals.group(1)) / conflicts_num
    else:
        average_len = 0


    reduction = re.search("(\d+\.\d+) % deleted,\s+", output)
    if (reduction is not None):
        conflict_reduction = float(reduction.group(1))
    else:
        conflict_reduction = 0

    i_uip_reduction = re.search("(\d+\.\d+) % deleted by i-uip", output)
    if (i_uip_reduction is not None):
        conflict_i_uip_reduction = float(i_uip_reduction.group(1))
    else:
        conflict_i_uip_reduction = 0
    return  conflicts_num, average_len, conflict_reduction, conflict_i_uip_reduction

def get_memory_use(output):
    mem_use = re.search("Memory used\s+:\s+(\d+\.\d+)\s+MB", output)
    if mem_use is not None:
        return mem_use.group(1)
    else:
        return "NAN"

def run_problem_and_write_result(solver, invocation, problem, fp, lock):
    result = run_problem(solver, invocation,problem)
    lock.acquire()
    try:
        fp.write("%s,%s,%s, %s,%s, %s, %s, %s, %s, %s ,%s, %s\n" % (
        result.problem, result.result, result.elapsed, result.conflict_size, result.average_len, result.reduction,
        result.i_uip_reduction, result.mem_use, result.core_clause, result.lbd , result.attempt_rate, result.success_rate))
    finally:
        lock.release()


def run_problem(solver, invocation, problem):
    # pass the problem to the command
    command = "%s %s" %(invocation, problem)
    # get start time
    start = datetime.datetime.now().timestamp()
    # run command
    process = subprocess.Popen(
        command,
        shell      = True,
        stdout     = subprocess.PIPE,
        stderr     = subprocess.PIPE,
        preexec_fn = os.setsid
    )
    # wait for it to complete
    try:
        process.wait(timeout=TIMEOUT)
    # if it times out ...
    except subprocess.TimeoutExpired:
        # kill it
        print('TIMED OUT:', repr(command), '... killing', process.pid, file=sys.stderr)
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        # set timeout result
        elapsed = TIMEOUT
        output  = TIMEOUT_RESULT
        average_conflict_c_size = "NAN";
        memory_use = "NAN"
        reward_score = "NAN"
        avg_LBD_score = "NAN"

    # if it completes in time ...
    else:
        # measure run time
        end     = datetime.datetime.now().timestamp()
        elapsed = end - start
        # get result
        stdout = process.stdout.read().decode("utf-8")
        stderr = process.stderr.read().decode("utf-8")
        output = output2result(problem, stdout + stderr)
        average_conflict_c_size, average_len, reduction, i_uip_reduction  = get_average_conflict_cluase_size(stdout)
        memory_use = get_memory_use(stdout)
        core_clause = get_core_clause(stdout)
        attempt_rate, success_rate = get_attempt_rate(stdout)
        lbd = get_lbd(stdout)
        #reward_score = get_reward_score(stdout)
        #avg_LBD_score = get_avg_LBD(stdout)
    # make result
    result = Result(
        problem  = problem.split("/", 2)[2],
        result   = output,
        elapsed  = elapsed,
        conflict_size = str(average_conflict_c_size),
        average_len = str(average_len),
        reduction = str(reduction),
        i_uip_reduction = str(i_uip_reduction),
        mem_use = memory_use,
        core_clause = str(core_clause),
        attempt_rate = attempt_rate,
        success_rate = success_rate,
        lbd = str(lbd)
    )
    return result


def run_solver(args, single_solver = False):
    solver   = args[0]
    command  = args[1]
    problems = args[2]
    problem_sets = args[3]
    finished_instances =[]

    for problem_set in problem_sets:

        problem_set_name =  problem_set[10:]
        if (len(problem_set_name) > 0):
            filename = "%s/%s_%s.csv" % (RESULTS_DIR, solver, problem_set[10:] )
            exists = os.path.isfile(filename)
            if (exists):
                file_data = np.genfromtxt(filename, delimiter=',', dtype=None, encoding=None,
                              names=["Instance", "Result", "Time", "conflict_size", "average_len", "reduction", "i_uip_reduction",  "mem_use", "core_clause", "lbd"],
                              skip_header=1)
                op = 'a+'
                finished_instances = [problem_set + '/' + item[0] for item in file_data]
            else:
                op = 'w+'
            with open(filename, op, buffering=1) as fp:
                if not exists:
                    fp.write(CSV_HEADER)
                if single_solver:
                    lock = multiprocessing.Lock()
                    number_processes = int(multiprocessing.cpu_count() /2) -1
                    pool = ThreadPool(number_processes)
                    tasks = []
                    for problem in problems:
                        if (problem.startswith(problem_set)) and problem not in finished_instances:
                            tasks.append((solver, command, problem, fp, lock))
                    results = pool.starmap_async(run_problem_and_write_result, tasks)
                    pool.close()
                    pool.join()
                else:
                    for problem in problems:
                        if (problem.startswith(problem_set) ) and problem not in finished_instances:
                            result = run_problem(solver, command, problem)
                            fp.write("%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" % (result.problem, result.result, result.elapsed, result.conflict_size, result.average_len, result.reduction, result.i_uip_reduction, result.mem_use, result.core_clause, result.lbd,
                                                                                  result.attempt_rate, result.success_rate))


def signal_handler(signal, frame):
    print("KILLING!")
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)


def main():
    signal.signal(signal.SIGTERM, signal_handler)
    problems = glob.glob(PROBLEMS, recursive=True)
    problem_sets = glob.glob(PROBLEMS_SET, recursive=False)
    print(len(problems))
    
    args = [[solver, command, problems, problem_sets] for solver, command in SOLVERS.items()]
    try:
        if len(args) == 1:
            arg = args[0]
            run_solver(arg, single_solver=True)
        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor.map(run_solver, args)
    except KeyboardInterrupt:
        print('Interrupted!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception as e:
        print (e)


if __name__ == '__main__':
    main()
