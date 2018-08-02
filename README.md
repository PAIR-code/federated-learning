# Federated Learning in TensorFlow.js

This is the parent repository for an (experimental and probably only demo-ready) implementation of [Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html) in [Tensorflow.js](https://js.tensorflow.org/).

## Basic Usage

On the server (NodeJS) side:

```js
import * as http from 'http';
import * as federated from 'tfjs-federated-learning-server';

const INITIAL_MODEL = 'file:///initial/model.json';
const httpServer = http.createServer();
const fedServer = federated.Server(httpServer, INITIAL_MODEL);

fedServer.onNewVersion((model, oldVersion, newVersion) => {
  console.log(`updated model from ${oldVersion} to ${newVersion}`);
});

fedServer.setup().then(() => {
  httpServer.listen(8080);
})
```

On the client (browser) side:

```js
import * as federated from 'tfjs-federated-learning-client';

const SERVER_URL = 'https://federated.learning.server';
const client = federated.Client(SERVER_URL);

client.onNewVersion((model, oldVersion, newVersion) => {
  console.log(`updated model from ${oldVersion} to ${newVersion}`);
});

client.setup().then((model) => {
  // make predictions!
  model.predict(x).then((yhat) => {
    yhat.print();
  });

  // train (and asynchronously update the server)!
  model.fit(x, y);
});
```

## Advanced Usage

See specific [server](./src/server/README.md) and [client](./src/client/README.md) docs.
