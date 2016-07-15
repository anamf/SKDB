This software is a minimum functional part of a major program, called Gigal
 (from Gigabyte learners), that aims to be an open source system for 
classification learning from very large data.

Since Gigal has not yet been released, we ask the reviewers to use this 
piece of software only as supporting material of the paper submitted, and 
not redistribute it. 

INSTRUCTIONS FOR COMPILATION: (a makefile is included with the code)

>> make gigal

EXAMPLE OF USAGE:

Generic:
>> ./gigal <metafile> <trainingfile> [-p<posClassName>] [<test method args>] -l<learner> [<learner args>] 

selective KDB:
>> ./gigal ../data/poker-hand.pmeta ../data/poker-hand.pdata -x -v2 -lkdb-Selective -k5

k-selective KDB:
>> ./gigal ../data/poker-hand.pmeta ../data/poker-hand.pdata -x -v2 -lkdb-Selective -selectiveK -k5

k-selective KDB with MCC as objective function:
>> ./gigal ../data/poker-hand.pmeta ../data/poker-hand.pdata -x -v2 -lkdb-Selective -selectiveK -selectiveMCC -k5

KDB:
>> ./gigal ../data/poker-hand.pmeta ../data/poker-hand.pdata -x -v2 -lkdb

AODE:
>> ./gigal ../data/poker-hand.pmeta ../data/poker-hand.pdata -x -v2 -laode

To test on originally numeric datasets (with mdl discretization):
>> ./gigal ../data/numeric.pmeta ../data/numeric.pdata -dmdl -x -v2 -laode
