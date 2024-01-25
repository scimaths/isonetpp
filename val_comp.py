import os

path = "./"
directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("gmn_match_hinge")]

run = '101'
for directory in directories:
    logspath = os.path.join(directory, 'logDir')
    logs = sorted(os.listdir(logspath))
    for log in logs:
        logfile = logspath + '/' + log
        print(logfile)
        with open(logfile, 'r') as f:
            text = f.readlines()
        current_val = None
        final_val = None
        reach = False
        for line in text:
            if 'map_score' in line:
                current_val = float(line.split(' ')[-3])
            if 'saving best validated' in line:
                final_val = current_val
            if f'Run: {run} train' in line:
                reach = True
                break
        if not reach:
            print('not reached')
        else:
            print(final_val)