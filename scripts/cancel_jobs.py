import subprocess
s = 359255 
e = s + 1200
for j in range(s, e+1) + range(e, s+1):
    subprocess.call("scancel " + str(j), shell=True)
