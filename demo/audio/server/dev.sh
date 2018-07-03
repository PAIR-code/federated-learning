#!/bin/bash

whereami=`pwd`
cd ../../../src/server
yarn publish-local
cd $whereami
./node_modules/.bin/yalc link federated-learning-server
yarn run ts-node server.ts
