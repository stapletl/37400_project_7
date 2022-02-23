#!

from calendar import c
from random import *
import numpy as np
import signal
import datetime
import argparse
import copy
import json
import copy

from memoization import cached

class TimeoutException(Exception):
    pass


def handle_maxSeconds(signum, frame):
    raise TimeoutException()


VERBOSE = True

# return True if clause has a true literal


def check_clause(clause, assignment):
    clause_val = False
    for i in clause:
        # assignment is 0-ref, variables 1-ref
        if np.sign(i) == np.sign(assignment[abs(i)-1]):
            clause_val = True
    return clause_val


def score(clauses, assignment):
    sum = 0
    for clause in clauses:
        if check_clause(clause, assignment):
            sum += 1
    return sum


def check(clauses, assignment):
    global VERBOSE

    if VERBOSE:
        print('Checking assignment {}'.format(assignment))
        print('score of assignment is {}'.format(score(clauses, assignment)))
    for clause in clauses:
        if not check_clause(clause, assignment):
            return clause
    print('Check succeeded!')
    return True


def random_walk(num_variables, clauses):
    print('Random walk search started')
    assignment = np.ones(num_variables)
    while True:
        if True == check(clauses, assignment):
            break
        var_to_flip = randint(1, num_variables)
        assignment[var_to_flip-1] *= -1
    print('Random walk search completed successfully')
    return assignment


def generate_solvable_problem(num_variables):
    global VERBOSE

    k = 3  # 3-SAT
    seed()

    # < 4.2 easy;  >4.2 usually unsolvable.  4.2 challenging to determine.
    clauses_per_variable = 4.2
    num_clauses = round(clauses_per_variable*num_variables)

    # this assignment will solve the problem
    target = np.array([2*randint(0, 1)-1 for _ in range(num_variables)])
    clauses = []
    for i in range(num_clauses):
        seeking = True
        while seeking:
            # choose k variables at random
            clause = sorted((sample(range(0, num_variables), k)))
            clause = [i+1 for i in clause]
            clause = [(2*randint(0, 1)-1)*clause[x]
                      for x in range(k)]  # choose their signs at random
            seeking = not check_clause(clause, target)
        clauses.append(clause)

    if VERBOSE:
        print('Problem is {}'.format(clauses))
        print('One solution is {} which checks to {}'.format(
            target, check(clauses, target)))

    return clauses


def backtrack_search(num_variables, clauses):
    print('Backtracking search started')

    def backtrack(assignment, i):
        # i is next variable number to try values for (1..numvariables)
        if i == num_variables+1:  # no more variables to try
            if check(clauses, assignment) == True:
                return assignment
            return None
        else:
            for val in [-1, 1]:
                # assignment is 0-ref, so assignment[x] stores variable x+1 value
                assignment[i-1] = val
                result = backtrack(assignment, i+1)
                if result != None:
                    return result
        return None

    assignment = np.array([1]*num_variables)
    result = backtrack(assignment, 1)
    print('Backtracking search completed successfully')
    return(result)


def better_random(num_variables, clauses):
    # random start state of -1s and 1s
    states = []
    state_scores = []
    for x in range(50):
        states.append(
            np.array([2*randint(0, 1)-1 for _ in range(num_variables)]))
        state_scores.append(score(clauses, states[x]))
    # return a state with the max score out of the 50 random states
    return states[state_scores.index(max(state_scores))]

'''
This function implements the (Davis-Putnam-Logemann-Loveland) DPLL algorithm. 
This algorithm picks variables, then backtracks based on if the clause is corrent. 
'''

@cached
def solve_dpll(num_variables, clauses, assignment=None):

    # print('clauses', clauses, 'len', len(clauses), 'assignment', assignment)

    def clean_clauses(alpha, clauses_arr):
        # alpha is pos or neg variable
        clauses_arr = [x for x in clauses_arr if alpha not in x] # delete clauses containing alpha
        for x in clauses_arr:
            if -alpha in x:  # remove !alpha from all clauses
                x.remove(-alpha)
        return clauses_arr

    if assignment is None:
        assignment = np.array([1]*num_variables)

    while sum(len(clause) == 1 for clause in clauses):  # repeat until there are no unit clauses

        for clause in clauses:

            if len(clause) == 1:  # find a unit clause
                # print('UNIT CLAUSE FOUND', clause)
                if clause[0] > 0:  # if unit clause is "True"
                    assignment[clause[0]-1] = 1
                    clauses = clean_clauses(clause[0], clauses)
                    break
                else:  # if unit clause is "False"
                    assignment[-clause[0]-1] = -1
                    clauses = clean_clauses(clause[0], clauses)
                    break

    if sum(len(clause) == 0 for clause in clauses):
        # if there is an empty clause this expression isn't correct
        return None

    if len(clauses) == 0:
        # if all clauses have been removed this expression is correct
        return assignment

    # pick the first variable in the first clause and test it
    alpha = abs(clauses[0][0])  # get the abs of the first variable

    assignment_true = assignment.copy()
    assignment_false = assignment.copy()
    clauses_true = copy.deepcopy(clauses)
    clauses_false = copy.deepcopy(clauses)

    if clauses[0][0] > 0:  # if chosen alpha is true
        assignment_true[alpha-1] = 1
        assignment_false[alpha-1] = -1
    else:  # if chosen alpha is false
        assignment_true[alpha-1] = -1
        assignment_false[alpha-1] = 1


    clauses_true = clean_clauses(clauses_true[0][0], clauses_true)
    clauses_false = clean_clauses(-clauses_false[0][0], clauses_false)

    #print("alpha to try is:", clauses[0][0])
    try1 = solve_dpll(num_variables, clauses_true, assignment_true)
    if try1 is not None:
        assignment = try1
        return assignment
    # print('try 1 failed')

    #print("alpha to try is:", -clauses[0][0])
    try2 = solve_dpll(num_variables, clauses_false, assignment_false)
    if try2 is not None:
        assignment = try2
        return assignment
    else:
        assignment = None

    return assignment

'''
This function implements a hill climb algorithm that picks a random starting state. 
The function then moves towards the higher score. If the algorithm finds the solution, exit. 
If the function cannot go to a higher, score, restart with another random start state. 
'''
def hillclimb(num_variables, clauses):
    print('Hill Climb search started')
    # random start state of -1s and 1s
    assignment = better_random(num_variables, clauses)
    while True:

        # if true we found a solution
        if True == check(clauses, assignment):
            break

        scores = [0] * num_variables
        for i in range(num_variables):
            assignment[i] *= -1
            scores[i] = score(clauses, assignment)
            assignment[i] *= -1

        # set the new state based on the max score
        newStateIndex = scores.index(max(scores))

        # if the new state has higher score go to that state
        if scores[newStateIndex] > score(clauses, assignment):
            assignment[newStateIndex] *= -1
        # else pick a new random starting point
        else:
            assignment = better_random(num_variables, clauses)

    print('Hill Climb seach completed successfully')
    return assignment

'''
This function implements a hill climb search algorithm with a tabu list. 
The tabu list, implemented as a dict, holds the steps that were run and prevents redundant runs
'''
def hillclimb_with_tabu(num_variables, clauses):
    print('Hill Climb with tabu search started')

    def better_random():
        # random start state of -1s and 1s
        states = []
        state_scores = []
        for x in range(50):
            states.append(
                np.array([2*randint(0, 1)-1 for _ in range(num_variables)]))
            state_scores.append(score(clauses, states[x]))
        # return a state with the max score out of the 50 random states
        return states[state_scores.index(max(state_scores))]

    assignment = better_random()
    tabuDict = {}
    while True:

        # if true we found a solution
        if True == check(clauses, assignment):
            break

        scores = [0] * num_variables
        for i in range(num_variables):
            assignment[i] *= -1
            scores[i] = score(clauses, assignment)
            assignment[i] *= -1

        # set the new state based on the max score
        newStateIndex = scores.index(max(scores))

        # if the new state has higher score go to that state
        if scores[newStateIndex] > score(clauses, assignment):
            assignment[newStateIndex] *= -1
            # if the new state has already been explored generate a new state
            if str(assignment) in tabuDict:
                assignment = better_random()
            # add the next assignment to the tabu table
            tabuDict[str(assignment)] = 1
        # else pick a new random starting point
        else:
            assignment = better_random()

    print('Hill Climb seach completed successfully')
    # print('len(tabuDict):', len(tabuDict))
    return assignment

'''
Stochastic hill climb search implements an algorithm that goes to the state with
the higher score. 
'''
def stochastic_hillclimb(num_variables, clauses):  # this one is very bad
    print('Stochastic Hill Climb search started')
    # random start state of -1s and 1s
    assignment = np.array([2*randint(0, 1)-1 for _ in range(num_variables)])
    while True:

        # if true we found a solution
        if True == check(clauses, assignment):
            break

        scores = [0] * num_variables
        for i in range(num_variables):
            assignment[i] *= -1
            scores[i] = score(clauses, assignment)
            assignment[i] *= -1

        # set the new state based on the max score
        currentScore = score(clauses, assignment)
        # list of all indecies with higher scores
        possibleStates = [i for i in range(
            len(scores)) if scores[i] > currentScore]

        # if a new state has higher score go to one of those states
        # ! I don't think this is the correct way or a good idea
        if len(possibleStates) > 0:
            randomBetterIndex = randint(0, len(possibleStates)+1)
            assignment[randomBetterIndex] *= -1
        # else pick a new random starting point
        else:
            assignment = np.array(
                [2*randint(0, 1)-1 for _ in range(num_variables)])

    print('Hill Climb seach completed successfully')
    return assignment

# timeout is provided in case your method wants to know

def hw7_submission(num_variables, clauses, timeout=None):
    #print('hw7_submission search started')
    assignment = solve_dpll(num_variables, clauses)
    #assignment = hillclimb_with_tabu(num_variables, clauses)
    #assignment = stochastic_hillclimb(num_variables, clauses)
    return assignment if assignment is not None else False


def solve_SAT(file, save, timeout, num_variables, algorithms, verbose):
    global VERBOSE

    VERBOSE = verbose

    if file != None:
        with open(file, "r") as f:
            [num_variables, timeout, clauses] = json.load(f)
        print('Problem with {} variables and timeout {} seconds loaded'.format(
            num_variables, timeout))
    else:
        clauses = generate_solvable_problem(num_variables)
        if timeout == None:
            timeout = round(60 * num_variables / 100)
        print('Problem with {} variables generated, timeout is {}'.format(
            num_variables, timeout))

    if save != None:
        with open(save, "w") as f:
            json.dump((num_variables, timeout, clauses), f)

    if 'hw7_submission' in algorithms:
        signal.signal(signal.SIGALRM, handle_maxSeconds)
        signal.alarm(timeout)
        startTime = datetime.datetime.now()
        try:
            result = "Timeout"
            result = hw7_submission(num_variables, clauses, timeout)
            print('Solution found is {}'.format(result))
            if not (True == check(clauses, result)):
                print('Returned assignment incorrect')
        except TimeoutException:
            print("Timeout!")
        endTime = datetime.datetime.now()
        seconds_used = (endTime - startTime).seconds
        signal.alarm(0)
        print('Search returned in {} seconds\n'.format(seconds_used))
    if 'backtrack' in algorithms:
        signal.signal(signal.SIGALRM, handle_maxSeconds)
        signal.alarm(timeout)
        startTime = datetime.datetime.now()
        try:
            result = "Timeout"
            result = backtrack_search(num_variables, clauses)
            print('Solution found is {}'.format(result))
            if not (True == check(clauses, result)):
                print('Returned assignment incorrect')
        except TimeoutException:
            print("Timeout!")
        endTime = datetime.datetime.now()
        seconds_used = (endTime - startTime).seconds
        signal.alarm(0)
        print('Search returned in {} seconds\n'.format(seconds_used))
    if 'random_walk' in algorithms:
        signal.signal(signal.SIGALRM, handle_maxSeconds)
        signal.alarm(timeout)
        startTime = datetime.datetime.now()
        try:
            result = "Timeout"
            result = random_walk(num_variables, clauses)
            print('Solution found is {}'.format(result))
            if not (True == check(clauses, result)):
                print('Returned assignment incorrect')
        except TimeoutException:
            print("Timeout!")
        endTime = datetime.datetime.now()
        seconds_used = (endTime - startTime).seconds
        signal.alarm(0)
        print('Search returned in {} seconds\n'.format(seconds_used))


def main():
    parser = argparse.ArgumentParser(
        description="Run stochastic search on a 3-SAT problem")
    parser.add_argument("algorithms", nargs='*',
                        help="Algorithms to try",
                        choices=['random_walk', 'hw7_submission', 'backtrack'])
    parser.add_argument(
        "-f", "--file", help="file name with 3-SAT formula to use", default=None)
    parser.add_argument(
        "-s", "--save", help="file name to save problem in", default=None)
    parser.add_argument(
        "-t", "--timeout", help="Seconds to allow (default based on # of vars)", type=int, default=None)
    parser.add_argument(
        "-n", "--numvars", help="Number of variables (default 20)", type=int, default=20)
    parser.add_argument(
        "-v", "--verbose", help="Whether to print tracing information", action="store_true")

    args = parser.parse_args()
    file = args.file
    save = args.save
    timeout = args.timeout
    num_variables = args.numvars
    algorithms = args.algorithms
    verbose = args.verbose

    if (file != None and (timeout != None or num_variables != None)):
        print('\nUsing input file, any command line parameters for number of variables and timeout will be ignored\n')
    solve_SAT(file, save, timeout, num_variables, algorithms, verbose)


if __name__ == '__main__':
    main()


# if you prefer to load the file rather than use command line
# parameters, use this section to configure the solver
#
# outfile = None # 'foo.txt'
# infile  = None # 'save500.txt'
# timeout = None #ignored if infile is present, will be set based on numvars if None here
# numvars = 70   #ignored if infile is present
# algorithms = ['random_walk', 'backtrack', 'hw7_submission']
# verbosity = False
# solve_SAT(infile, outfile, timeout, numvars, algorithms, verbosity)
