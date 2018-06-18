set -e
set -u
N_CLIENTS=$1
SPLIT_SIZE=$2

DATASET_UPPER=$((N_CLIENTS * SPLIT_SIZE))

function finish {
  rm -rf data
  pkill -P $$
}

trap finish EXIT

mkdir -p data

for i in seq 0 $SPLIT_SIZE $DATASET_UPPER;
do
  ../node_modules/.bin/ts-node ./experiment_mnist_transfer_learning.ts $i &
done

wait
finish()
