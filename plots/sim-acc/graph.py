with open("plots/sim-acc/sim_acc.txt", "r") as f:
    lines = f.readlines()

local_windows = []
factors = []

for line in lines[2:]:
    local_windows.append(float(line.split("\t")[1].strip()))
    factors.append(float(line.split("\t")[3].strip()))

print(local_windows)
print(factors)
