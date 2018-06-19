from itertools import product
import os

CLIENTS=[2, 11, 22, 33, 101, 1001]
EXAMPLES=[5, 10, 30, 40]
AVG_EVERY=[10, 100, 200]
SYNC_EVERY=[1, 3, 5]

for (nclients, nexamples, nsync, navg) in product(CLIENTS, EXAMPLES, SYNC_EVERY, AVG_EVERY):
  if nclients * nexamples < nsync or nclients * nexamples < navg:
    continue

  log_filename = "logs/{}_{}_{}_{}.txt".format(nclients, nexamples, nsync, navg)
  print("running", (nclients, nexamples, nsync, navg), log_filename)
  os.system("bash ./reset_and_launch.sh {} {} {} {} > {}".format(nclients, nexamples, nsync, navg, log_filename))
