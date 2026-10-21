"""Error table, mean and standard deviation for Project 5 (Tasks 2 and 4).

Usage:  python3 stats.py task2.csv   (or task4.csv)
"""
import csv
import math
import statistics
import sys

TARGET = 1.0  # metres

path = sys.argv[1] if len(sys.argv) > 1 else 'task2.csv'
ruler_err, odom_err = [], []

print(f'{"run":>3} {"ruler_m":>8} {"odom_m":>8} {"ruler_err":>10} {"odom_err":>10}')
with open(path) as f:
    for row in csv.DictReader(f):
        if not row['ruler_m'] or not row['end_y']:
            continue  # run not filled in yet
        ruler = float(row['ruler_m'])
        odom = math.hypot(float(row['end_x']) - float(row['start_x']),
                          float(row['end_y']) - float(row['start_y']))
        ruler_err.append(ruler - TARGET)
        odom_err.append(odom - TARGET)
        print(f'{row["run"]:>3} {ruler:8.4f} {odom:8.4f} '
              f'{ruler_err[-1]:+10.4f} {odom_err[-1]:+10.4f}')

if len(ruler_err) < 2:
    sys.exit('Need at least 2 filled-in runs.')
print()
for name, errs in (('Actual (ruler)', ruler_err), ('Odometry', odom_err)):
    print(f'{name:15} mean = {statistics.mean(errs):+.4f} m   '
          f'std dev = {statistics.stdev(errs):.4f} m')
