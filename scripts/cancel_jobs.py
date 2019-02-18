import subprocess
s = 439921 
e = s + 2800
for j in list(range(s, e+1)) + list(range(e, s+1)):
    subprocess.call("scancel " + str(j), shell=True)
