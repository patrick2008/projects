import numpy as np
import matplotlib.pyplot as plt
from random import sample
import math

"""
Sudoku board initializer
Credit: https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python
"""
def generate(n, num_clues):
    # Generate a sudoku problem of order n with "num_clues" cells assigned
    # Return dictionary containing clue cell indices and corresponding values
    # (You do not need to worry about components inside returned dictionary)
    N = range(n)

    rows = [g*n+r for g in sample(N,n) for r in sample(N,n)]
    cols = [g*n+c for g in sample(N,n) for c in sample(N,n)]
    nums = sample(range(1,n**2+1), n**2)

    S = np.array([[nums[(n*(r%n)+r//n+c)%(n**2)] for c in cols] for r in rows])
    indices = sample(range(n**4), num_clues)
    values = S.flatten()[indices]

    mask = np.full((n**2, n**4), True)
    mask[:, indices] = False
    i, j = np.unravel_index(indices, (n**2,n**2))

    for c in range(num_clues):
        v = values[c]-1
        maskv = np.full((n**2, n**2), True)
        maskv[i[c]] = False
        maskv[:,j[c]] = False
        maskv[(i[c]//n)*n:(i[c]//n)*n+n,(j[c]//n)*n:(j[c]//n)*n+n] = False
        mask[v] = mask[v] * maskv.flatten()

    return {'n':n, 'indices':indices, 'values':values, 'valid_indices':mask}


def display(problem):
    # Display the initial board with clues filled in (all other cells are 0)
    n = problem['n']
    empty_board = np.zeros(n**4, dtype=int)
    empty_board[problem['indices']] = problem['values']
    print("Sudoku puzzle:\n", np.reshape(empty_board, (n**2,n**2)), "\n")


def initialize(problem):
    # Returns a random initial sudoku board given problem
    n = problem['n']
    S = np.zeros(n**4, dtype=int)
    S[problem['indices']] = problem['values']

    all_values = list(np.repeat(range(1,n**2+1), n**2))
    for v in problem['values']:
        all_values.remove(v)
    all_values = np.array(all_values)
    np.random.shuffle(all_values)

    indices = [i for i in range(S.size) if i not in problem['indices']]
    S[indices] = all_values
    S = S.reshape((n**2,n**2))

    return S


def successors(S, problem):
    # Returns list of all successor states of S by swapping two non-clue entries
    mask = problem['valid_indices']
    indices = [i for i in range(S.size) if i not in problem['indices']]
    succ = []

    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            s = np.copy(S).flatten()
            if s[indices[i]] == s[indices[j]]: continue
            if not (mask[s[indices[i]]-1, indices[j]] and mask[s[indices[j]]-1, indices[i]]): continue
            s[indices[i]], s[indices[j]] = s[indices[j]], s[indices[i]]
            succ.append(s.reshape(S.shape))

    return succ


"""
WRITE THIS FUNCTION
"""
def num_errors(S):
    # Given a current sudoku board state (2d NumPy array), compute and return total number of errors
    # Count total number of missing numbers from each row, column, and non-overlapping square blocks
    error = 0
    n2 = len(S)
    for i in S:
        row_track = {}
        for j in i:
            row_track[j]=1
        error += n2 - len(row_track)
    for j in range(n2):
        column_track = {}
        for i in range(n2):
            column_track[S[i][j]] = 1
        error += n2 - len(column_track)

    n = int(math.sqrt(n2))
    for sub_row in range(0,n2,n):
        for sub_column in range(0,n2,n):
            n_track = {}
            for i in range(sub_column, sub_column+n):
                for j in range(sub_row, sub_row+n):
                    n_track[S[i][j]]=1
            error+= n2-len(n_track)
            #print(error)
    return error


"""
WRITE THIS FUNCTION
"""
def hill_climb(problem, max_sideways=0, max_restarts=0):
    # Given: Sudoku problem and optional max sideways moves and max restarts parameters
    # Return: Board state solution (2d NumPy array), list of errors in each iteration of hill climbing search
    start = initialize(problem)
    board = start
    errors = num_errors(board)
    error_list = [errors]
    side = 0
    resets = 0

    while errors>0:
        succ = successors(board, problem)
        min_val = 2000000000
        mins = []
        for i in range(len(succ)):
            curr_error = num_errors(succ[i])
            if curr_error == min_val:
                mins.append(i)
            elif curr_error < min_val:
                min_val = curr_error
                mins = [i]
        if min_val >= errors:
            if min_val==errors:
                side+=1
                if side >= max_sideways:
                    resets+=1
                    if resets >= max_restarts:
                        break
                    board = start
                    continue
            else:
                resets+=1
                if resets >= max_restarts:
                    break
                board = start
                continue
        else:
            side = 0
        errors = min_val
        error_list.append(errors)
        board = succ[sample(mins, 1)[0]]
            
    return board, error_list


if __name__ == '__main__':
    n = 3
    clues = 40
    # success = float(0)
    # average = float(0)
    # for i in range(100):
    #     problem = generate(n, clues)
    #     sol, errors = hill_climb(problem,10,10)
    #     if errors[-1]==0:
    #         success+=1
    #     average+=errors[-1]
    # success/=100
    # average/=100
    # print(success)
    # print(average)

    problem = generate(n, clues)
    display(problem)
    sol, errors = hill_climb(problem,40,20)
    print("Solution:\n", sol)
    print(errors[-1])
    plt.plot(errors)
    plt.show()
