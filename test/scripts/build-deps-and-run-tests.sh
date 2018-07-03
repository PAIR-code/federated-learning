#!/bin/bash

echo '***** locally publish federated-learning-server *****'
cd ../src/server
yarn publish-local
echo ''
echo '***** locally publish federated-learning-client *****'
cd ../client
yarn publish-local
echo ''
echo '***** link local libraries *****'
cd ../../test
./node_modules/.bin/yalc link federated-learning-client
./node_modules/.bin/yalc link federated-learning-server
echo ''
echo '***** run tests *****'
ts-node node_modules/jasmine/bin/jasmine --config=jasmine.json
