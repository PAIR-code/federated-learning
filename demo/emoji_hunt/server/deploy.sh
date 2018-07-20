whereami=`pwd`

DEV_ENV=${DEV:false}

if [ ! -e cert.pem ]
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
  yarn run parcel watch index.html &
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

USE_OAUTH=1 SSL_KEY=key.pem SSL_CERT=cert.pem PORT=$PORT $TSNODE index.ts
