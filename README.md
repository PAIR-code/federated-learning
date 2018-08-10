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

## A Note about Privacy

Federated learning is considered "private" because clients never send their data to the server, only updated weights. In an ideal world,
1. the server should not be able to meaningfully reconstruct client data from weight updates
2. one client should not be able to reliably detect the presence of another client from model updates

To help achieve (1), we can:
- increase `numExamplesPerUpdate` (averages together more individual data points before computing the update)
- increase `weightNoiseStddev` (adds noise to the update)
- possibly change `epochs`, `batchSize`, and `learningRate` (clear that modifying these affects reconstructability, but unclear exactly _how_)
- implement secure aggregation (encrypt update and only allow decryption after averaging with many other users)
- modify the model architecture (different layer updates contain different amounts of information about inputs)

To help achieve (2), we can:
- increase `numUpdatesPerVersion` (average together more updates before computing the new version)
- implement new features that limit the contributions of individual clients to the new version
