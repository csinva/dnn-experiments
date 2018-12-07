import subprocess
s = 375156 
e = s + 2800
for j in range(s, e+1) + range(e, s+1):
    subprocess.call("scancel " + str(j), shell=True)
