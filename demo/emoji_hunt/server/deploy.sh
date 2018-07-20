whereami=`pwd`

DEV_ENV=${DEV:-false}
SSL_CERT=${SSL_CERT:-cert.pem}
SSL_KEY=${SSL_KEY:-key.pem}

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

USE_OAUTH=1 SSL_KEY=$SSL_KEY SSL_CERT=$SSL_CERT PORT=$PORT $TSNODE index.ts
