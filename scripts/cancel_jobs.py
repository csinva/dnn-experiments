import subprocess
s = 350170 
e = s + 300
for j in range(s, e+1) + range(e, s+1):
    subprocess.call("scancel " + str(j), shell=True)
