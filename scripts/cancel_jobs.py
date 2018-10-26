import subprocess
s = 337432 
e = s + 100
for j in range(s, e+1) + range(e, s+1):
    print("scancel " + str(j))
    subprocess.call("scancel " + str(j), shell=True)
