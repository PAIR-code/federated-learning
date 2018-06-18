set -e
set -u
N_CLIENTS=$(($1 - 1))
SPLIT_SIZE=$2
SYNC_EVERY=$3

DATASET_UPPER=$((N_CLIENTS * SPLIT_SIZE))

DATADIR=$(mktemp -d)

function finish {
  rm -rf $DATADIR
  pkill -P $$
}

trap finish EXIT

../node_modules/.bin/ts-node ./experiment_mnist_server.ts $DATADIR &

for i in `seq 0 $SPLIT_SIZE $DATASET_UPPER`;
do
  TF_CPP_MIN_LOG_LEVEL=1 ../node_modules/.bin/ts-node ./experiment_mnist_transfer_learning.ts $i $SPLIT_SIZE $SYNC_EVERY &
done

wait
finish()
