import os
import random
rootDir = "/home/ai/BVe-G/Images/BV2_Total"
fileSet = set()
f = open('bmppath.txt', 'w')
for dir_, _, files in os.walk(rootDir):
    for fileName in files:
        relDir = os.path.relpath(dir_, rootDir)
        #print(relDir)
        relFile = os.path.join(rootDir, relDir, fileName)
        f.write("\n%s" % (relFile))
        #print(relFile)
        fileSet.add(relFile)

with open("bmppath.txt") as f:
    lines = f.readlines()
random.shuffle(lines)
with open("bmppath.txt", "w") as f:
    f.writelines(lines)