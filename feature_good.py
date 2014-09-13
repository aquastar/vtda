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
    cc = {}
    cc[0] = 0
    for ttt in tt:
        if cc.has_key(tt[ttt]):
            cc[tt[ttt]] += 1
        else:
            cc[tt[ttt]] = 1
    for sss in freq:
         if tt.has_key(sss):
             pass
         else:
             cc[0] += 1
             tt[sss] = 0
    for s in freq:
        if tt.has_key(s):
            if cc.has_key(tt[s]+1):
                print "%5.5f," % ((float(tt[s]+1)/float(lcnt))*(float(cc[tt[s]+1])/float(cc[tt[s]]))),
            else:
                print "%5.5f," % 0,
        else:
            print "%5.5f," % ( (float(1)/float(lcnt)) * (float(cc[tt[s]+1])/float(cc[tt[s]])) ),
    print "0"

file_prefix = "./1661_"+str(sp)+"/"
files = os.listdir(file_prefix)
# for chi-square or info gain
for f in files:
    print "1661"+str(f)+",",
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
    cc = {}
    cc[0] = 0
    for ttt in tt:
        if cc.has_key(tt[ttt]):
            cc[tt[ttt]] += 1
        else:
            cc[tt[ttt]] = 1
    for sss in freq:
         if tt.has_key(sss):
             pass
         else:
             cc[0] += 1
             tt[sss] = 0
    for s in freq:
        if tt.has_key(s):
            if cc.has_key(tt[s]+1):
                print "%5.5f," % ((float(tt[s]+1)/float(lcnt))*(float(cc[tt[s]+1])/float(cc[tt[s]]))),
            else:
                print "%5.5f," % 0,
        else:
            print "%5.5f," % ( (float(1)/float(lcnt)) * (float(cc[tt[s]+1])/float(cc[tt[s]])) ),
    print "0"



