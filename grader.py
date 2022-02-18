# This is a sample stochastic search evaluation program you can play with
#
# Usage:
#   -- have a file hw7_submission.py with a function hw7_submission(num_variables, clauses) from the distributed template
#   -- There are two modes of operation: problem generation and problem evaluation.
#
# Problem generation:
#   python3 grader.py -n <size> <size> <size> ... <size> - <directory>
#      will generate problems of these sizes and store them in the specified directory, or ./evaluation_problems if not specified
#
# Problem evaluation:
#   python3 grader.py -d <directory> [-c <comment>]
#      will run hw7_submission() on each problem in the directory specified, or ./evaluation_problems if not specified.
#      will write the results to stdout but also to the file 'log' in the working directory, with the comment <comment> added
#      Problems in <directory> must be in files following the naming convention used in problem generation above.
#
# We provided a sample set of problems for you to evaluate on similar to what we might use.
#
# MINIMUM STANDARD:
#       The minimum standard to not have grade penalties from this project can be estimated as achieving <= 0.5 on the following test.
#       (Our actual test will use this size of problems but not these exact problems)
#   python3 grader.py -d minimum
#
# Score is the average percent of the allowed time needed across the set of problems.  The problem evaluator will stop if you
# timeout on two problems in a row, running smallest to largest, and assume you will timeout on all larger problems.
#
# this is a work in progress, please provide feedback

from sys import maxsize
from random import sample
from random import randint
import numpy as np
import signal, datetime
import argparse
import copy
import json
import hw7_submission as submission
from datetime import datetime as dt
from os import listdir
from os import mkdir
from os.path import isfile, join

class TimeoutException(Exception):
    pass

def handle_maxSeconds(signum, frame):
    raise TimeoutException()

VERBOSE=False


def generate_solvable_problem(num_variables,num_clauses,verbose):
    k=3 # 3-SAT

    # this assignment must solve the problem
    target = np.array([2*randint(0,1)-1 for _ in range(num_variables)]) 
    clauses=[]
    for i in range(num_clauses):
        seeking = True
        while seeking:
            clause=sorted((sample(range(0,num_variables),k))) # choose k variables at random 
            clause=[i+1 for i in clause]
            clause=[(2*randint(0,1)-1)*clause[x] for x in range(k)] # choose their signs at random
            seeking = not check_clause(clause,target)
        clauses.append(clause)

    if verbose:
        print('Problem is {}'.format(clauses))
        print('One solution is {} which checks to {}'.format(target,check(clauses,target)))
        
    return clauses

def write_SAT_problem_file(directory,num_variables,verbose):
    clauses_per_variable = 4.2  # < 4.2 easy;  >4.2 usually unsolvable.  4.2 challenging to determine.
    num_clauses=round(clauses_per_variable*num_variables)
    clauses = generate_solvable_problem(num_variables,num_clauses,verbose)
    timeout = round(60 * num_variables / 100)
    print('Problem with {} variables generated, timeout is {}'.format(num_variables,timeout))
    try:
        mkdir(f'{directory}')
    except OSError:
        None
    with open(f'{directory}/{num_variables}_variables.txt',"w") as f:
        json.dump((num_variables,timeout,clauses),f)

def read_SAT_problem(directory,num_variables):
    with open(f'./{directory}/{num_variables}_variables.txt',"r") as f:
        [num_variables,timeout,clauses] = json.load(f)
    print('Problem with {} variables and timeout {} seconds loaded'.format(num_variables,timeout))
    return (num_variables,timeout,clauses)

def solve_SAT_problem(num_variables,timeout,clauses,verbose):
    global VERBOSE
    
    VERBOSE=verbose
    submission.VERBOSE = verbose
    
    unsound = False
    signal.signal(signal.SIGALRM, handle_maxSeconds)
    signal.alarm(timeout)
    startTime = datetime.datetime.now()
    try:
        result=submission.hw7_submission(num_variables,clauses)
        print('Solution found is {}'.format(result))
        if not (True == check(clauses,result)):
            unsound = True
            print('Returned assignment incorrect')
    except TimeoutException:
        result="Timeout"
        print("Timeout!")
    endTime = datetime.datetime.now()
    seconds_used = (endTime - startTime).seconds
    if unsound:
        seconds_used = maxsize
    print('Search returned in {} seconds\n'.format(seconds_used))
    signal.alarm(0)
    return result, seconds_used

def check_clause(clause, assignment):
    # print('checking clause {} against assignment {}'.format(clause,assignment))
    clause_val = False
    for i in clause:
        if np.sign(i)==np.sign(assignment[abs(i)-1]): #assignment is 0-ref, variables 1-ref
            clause_val=True
    # print('--check came out {}'.format(clause_val))
    return clause_val

def check(clauses,assignment):
    global VERBOSE
    
    if VERBOSE:
        print('Checking assignment {}'.format(assignment))
    for clause in clauses:
        if not check_clause(clause, assignment):
            return clause
    print('Check succeeded!')
    return True
       
def main():
    parser = argparse.ArgumentParser(description="Generate a problem set of varying sizes or evaluate a hw7_submission function on a problem set")
    parser.add_argument("-v", "--verbose", help="Whether to print tracing information", action="store_true")
    parser.add_argument("-n", "--numvars", nargs='+', help="sequence of variable sizes to generate", type=int, default=None)
    parser.add_argument("-d", "--directory", help="directory from which to run problems", default="evaluation_problems")
    parser.add_argument("-c", "--comment", help="comment for log file line", default="")

    args = parser.parse_args()
    verbose = args.verbose
    directory = args.directory
    comment = args.comment

    result_dict = {}
    timeout_cnt = 0
    run = True
    
    if args.numvars: # generate a set with new num_variable
        num_variables_lst = args.numvars
        for num_variables in num_variables_lst:
            write_SAT_problem_file(directory,num_variables,verbose)
    else:
        onlyfiles = sorted([f for f in listdir(directory) if (isfile(join(directory, f)) and (not f.startswith('.')))],\
                        key=lambda x:int(x.split("_")[0]))
        print("Files list :", onlyfiles)
        
        for f in onlyfiles:
            num_variables = int(f.split('_')[0])
            (num_variables, timeout, clauses) = read_SAT_problem(directory,num_variables)
            if timeout_cnt>1:
                time_used=timeout
            else:
                result, time_used = solve_SAT_problem(num_variables,timeout,clauses,verbose)#use the files in the given evaluation_problems
                if time_used > timeout and time_used != maxsize: # maxsize means unsound answer returned
                    print("Timer failed")
                    time_used = timeout
                    result="Timeout"
                if type(result)==str and result == "Timeout":
                    timeout_cnt += 1
                else:
                    timeout_cnt = 0
            result_dict[num_variables] = (time_used,timeout)

        print("Results :", result_dict)
        average_score = sum(map(lambda v:float(v[0])/v[1],result_dict.values()))/len(result_dict)
        print(f'Average Score : {average_score:.3f} --- {comment} ')
        with open(f'./log',"a") as f:
            n = dt.now()
            f.write(n.strftime("%x%X")+f' Average score: {average_score:.3f} from {result_dict} --- {comment}\n')

if __name__ == '__main__':
    main()


