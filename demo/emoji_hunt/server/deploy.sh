#!/bin/bash
# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


whereami=`pwd`

DEV_ENV=${DEV:-false}
SSL_CERT=${SSL_CERT:-cert.pem}
SSL_KEY=${SSL_KEY:-key.pem}
USE_OAUTH=${USE_OAUTH:-1}

if [ ! -e $SSL_CERT ]
then
  echo "generating self-signed cert..."
  openssl req -x509 -newkey rsa:4096 -keyout key.pem \
      -out cert.pem -days 365 \
      -nodes \
      -sha256 \
      -subj '/CN=localhost'
fi

cd ../client

if [ "$DEV_ENV" = true ]
then
  yarn run parcel watch --no-hmr index.html &
else
  yarn build
fi

cd $whereami

TSNODE=ts-node

if [ "$DEV_ENV" = true ]
then
  TSNODE=ts-node-dev
fi

PORT=${PORT:-3000}

USE_OAUTH=$USE_OAUTH SSL_KEY=$SSL_KEY SSL_CERT=$SSL_CERT PORT=$PORT $TSNODE index.ts
