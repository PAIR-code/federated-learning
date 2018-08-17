cd src/client
npm version $1 --allow-same-version
yarn build-npm
cd ../mock_server
npm version $1 --allow-same-version
yarn build-npm
cd ../server
npm version $1 --allow-same-version
yarn build-npm
cd ../../
git tag v$1
