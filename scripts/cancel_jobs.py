import subprocess
s = 323766
e = s + 20
for j in range(s, e+1) + range(e, s+1):
    print("scancel " + str(j))
    subprocess.call("scancel " + str(j), shell=True)
