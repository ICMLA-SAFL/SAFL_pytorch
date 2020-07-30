import sys
voc = []
file = open('vocab.txt', "r")
for line in file:
    line = line.split('\n')
    voc.append(line[0])
print(voc)