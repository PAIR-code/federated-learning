#!/bin/bash

whereami=`pwd`
cd ../../../src/client
yarn publish-local
cd $whereami
yarn run yalc link federated-learning-client
yarn run cross-env NODE_ENV=development parcel index.html --no-hmr --open
