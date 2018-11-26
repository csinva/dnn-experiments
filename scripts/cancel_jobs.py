import subprocess
s = 352103 
e = s + 600
for j in range(s, e+1) + range(e, s+1):
    subprocess.call("scancel " + str(j), shell=True)
