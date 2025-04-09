# (define (problem hanoi0)
#   (:domain myhanoi)
#   (:objects
#     peg1 peg2 peg3 d1 d2 d3 - OBJ
#     p1 p2 p3 - PEG
#   )
#   (:init
#    (smaller peg1 d1) (smaller peg1 d2) (smaller peg1 d3)
#    (smaller peg2 d1) (smaller peg2 d2) (smaller peg2 d3)
#    (smaller peg3 d1) (smaller peg3 d2) (smaller peg3 d3)
#    (smaller d2 d1) (smaller d3 d1) (smaller d3 d2)
#    (clear peg2) (clear peg3) (clear d1)
#    (on d3 peg1) (on d2 d3) (on d1 d2)
#    (obj-at-peg d1 p1)
#    (obj-at-peg d2 p1)
#    (obj-at-peg d3 p1)
#    (obj-at-peg peg1 p1)
#    (obj-at-peg peg2 p2)
#    (obj-at-peg peg3 p3)
#   )
#   (:goal (and (on d3 peg3) (on d2 d3) (on d1 d2)))
#   )

# Generate something like above
# Goal: all objects on peg3
# Initial state: objects can be at any peg
# Input arguments: number of disks, number of pegs, seed

import random
import os
import sys
import argparse

def generate_problem(num_disks, num_pegs, seed):
    random.seed(seed)
    PEGS = ["p{}".format(i) for i in range(1, num_pegs+1)]
    pegs = ["peg{}".format(i) for i in range(1, num_pegs+1)]
    disks = ["d{}".format(i) for i in range(1, num_disks+1)]

    objs_at_peg = [[] for _ in range(num_pegs)]
    objs_at_peg_goal = [[] for _ in range(num_pegs)]
    for i in range(num_pegs):
        objs_at_peg[i].append(pegs[i])
        objs_at_peg_goal[i].append(pegs[i])
    for i in reversed(range(num_disks)):
        r = 0
        objs_at_peg[r].append(disks[i])

        r = num_pegs - 1
        objs_at_peg_goal[r].append(disks[i])

    init_str = ""
    for i in range(num_pegs):
        for j in range(len(objs_at_peg[i])):
            init_str += "(obj-at-peg {} {}) ".format(objs_at_peg[i][j], PEGS[i])
    for i in range(num_pegs):
        init_str += "(clear {}) ".format(objs_at_peg[i][-1])
    for i in range(num_pegs):
        for j in range(len(objs_at_peg[i])-1):
            init_str += "(on {} {}) ".format(objs_at_peg[i][j+1], objs_at_peg[i][j])
    for i in range(num_pegs):
        for j in range(num_disks):
            init_str += "(smaller {} {}) ".format(pegs[i], disks[j])
    for i in range(num_disks):
        for j in range(i):
            init_str += "(smaller {} {}) ".format(disks[i], disks[j])

    goal_str = ""
    for i in range(num_pegs):
        for j in range(len(objs_at_peg_goal[i])-1):
            goal_str += "(on {} {}) ".format(objs_at_peg_goal[i][j+1], objs_at_peg_goal[i][j])

    return f"""(define (problem hanoi{num_disks})
    (:domain myhanoi)
    (:objects
        {" ".join(pegs)} {" ".join(disks)} - OBJ
        {" ".join(PEGS)} - PEG
    )
    (:init
        {init_str}
    )
    (:goal (and {goal_str}))
    )"""

def main():
    parser = argparse.ArgumentParser(description='Generate a Hanoi problem')
    parser.add_argument('-n', type=int, help='Number of disks')
    parser.add_argument('-p', type=int, help='Number of pegs')
    parser.add_argument('-s', type=int, help='Random seed')
    args = parser.parse_args()
    print(generate_problem(args.n, args.p, args.s))

if __name__ == '__main__':
    main()
