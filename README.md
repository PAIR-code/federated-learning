# Federated Learning in TensorFlow.js

This is the parent repository for an (experimental and probably only demo-ready) implementation of [Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html) in [Tensorflow.js](https://js.tensorflow.org/). Federated Learning is a method for training machine learning models in a distributed fashion. Although it involves a central server, that server never needs to see any data or even compute a gradient. Instead, _clients_ perform all of the inference and training locally (which they already do in Tensorflow.js), and just periodically send the server updated weights (rather than data). The server's only job is to aggregate and redistribute them, which means it can be extremely lightweight!

## Basic Usage

On the server (NodeJS) side:

```js
import * as http from 'http';
import * as federated from 'tfjs-federated-learning-server';

const INITIAL_MODEL_URL = 'file:///initial/model.json';
const httpServer = http.createServer();
const fedServer = new federated.Server(httpServer, INITIAL_MODEL_URL);

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

const SERVER_URL = 'https://federated.learning.server'; // points to server above
const client = new federated.Client(SERVER_URL);

client.onNewVersion((model, oldVersion, newVersion) => {
  console.log(`updated model from ${oldVersion} to ${newVersion}`);
});

client.setup().then(() => {
  // make predictions!
  const yhat = client.predict(x);

  // train (and asynchronously update the server)!
  client.fit(x, y);
});
```

## Advanced Usage

See specific [server](./src/server/README.md) and [client](./src/client/README.md) docs.
