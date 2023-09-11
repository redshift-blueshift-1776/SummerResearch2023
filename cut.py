fname = "gps230101g.002.txt"
fhand = open(fname)
fout = open('cutdata.txt', 'w')
i = 0
for line in fhand:
    if i % 3 == 0:
        fout.write(line)
        # print(line)
    if i > 15420:
        break
    i += 1
