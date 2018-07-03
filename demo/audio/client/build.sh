#!/bin/bash

whereami=`pwd`
cd ../../../src/client
yarn publish-local
cd $whereami
./node_modules/.bin/yalc link federated-learning-client
yarn run cross-env NODE_ENV=production parcel build index.html --no-minify --public-url ./
