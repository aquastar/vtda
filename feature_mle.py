__author__ = 'zz'
import os

freq = [
    "miss",
    "time",
    "good",
    "lady",
    "man",
    "day",
    "great",
    "sister",
    "thing",
    "thought",
    "friend",
    "dear",
    "make",
    "made",
    "sir",
    "young",
    "house",
    "give",
    "room",
    "father",
    "fanny",
    "hope",
    "long",
    "feeling",
    "mother",
    "emma",
    "letter",
    "moment",
    "mind",
    "felt",
    "family",
    "manner",
    "woman",
    "love",
    "found",
    "year",
    "home",
    "place",
    "word",
    "crawford",
    "jane",
    "eye",
    "elizabeth",
    "heard",
    "hour",
    "happy",
    "heart",
    "morning",
    "brother",
    "elinor"
]
sp = 80

file_prefix = "./31100_"+str(sp)+"/"
files = os.listdir(file_prefix)
# for chi-square or info gain

for f in files:
    print "31100"+str(f)+",",
    file = open(file_prefix + f, "r")
    tt = {}
    lcnt = 0
    for i in file:
        lcnt += 1
        tmp = i.strip("\n")
        if tmp in freq:
            if not tt.has_key(tmp):
                tt[tmp] = 1
            else:
                tt[tmp] += 1
    for s in freq:
        if tt.has_key(s):
            print "%5.5f," % (float(tt[s])/float(lcnt)),
        else:
            print "%5.5f," % 0,
    print "0"

file_prefix = "./1661_"+str(sp)+"/"
files = os.listdir(file_prefix)
# for chi-square or info gain
for f in files:
    print "31100"+str(f)+",",
    file = open(file_prefix + f, "r")
    tt = {}
    lcnt = 0
    for i in file:
        lcnt += 1
        tmp = i.strip("\n")
        if tmp in freq:
            if not tt.has_key(tmp):
                tt[tmp] = 1
            else:
                tt[tmp] += 1
    for s in freq:
        if tt.has_key(s):
            print "%5.5f," % (float(tt[s])/float(lcnt)),
        else:
            print "%5.5f," % 0,
    print "0"



