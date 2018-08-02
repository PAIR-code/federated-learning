# TensorflowJS Federated Learning Server

This library sets up a simple websocket-based server for transmitting and receiving
TensorflowJS model weights.

## Usage

### Basic

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

### Setting the Initial Model

```js
federated.Server(httpServer, tfModel);
federated.Server(httpServer, 'https://remote.server/tf-model.json');
federated.Server(httpServer, 'file:///my/local/file/tf-model.json');
federated.Server(httpServer, async () => {
  const model = await tf.loadModel('file:///transfer/learning/model.json');
  model.layers[0].trainable = false;
  return model;
});
federated.Server(httpServer, federatedServerModel); // see below
```

The simplest way to set up a `federated.Server` is to pass a [`tf.Model`](https://js.tensorflow.org/api/0.12.0/#class:Model). However, you can also pass a string that will be delegated to [`tf.loadModel`](https://js.tensorflow.org/api/0.12.0/#loadModel) (both `https?://` and `file://` URLs should work), or an asynchronous function that will return a `tf.Model`. The final option is to define your own `FederatedServerModel`, which has to implement various saving and loading methods. See its [documentation](#TODO) for more details.

Note that by default, different `tf.Model` versions will be saved as files in subfolders of `${__dirname}/federated-server-models/`. If you would like to change this directory, you can pass a `modelDir` configuration parameter, e.g. `federated.Server(httpServer, model, { modelDir: '/mnt/my-vfs' })`.

### Setting Hyperparameters

```js
federated.Server(httpServer, model, {
  // These hyperparams only affect the server
  updatesPerVersion: 20, // server merges every 20 client weight updates
  weightAggregator: 'mean' // how to merge weights (only mean supported now)
  // These get broadcast to clients
  clientHyperparams: {
    examplesPerUpdate: 10, // client computes weight updates every 10 examples
    learningRate: 0.01, // client takes SGD steps of size 0.01
    epochs: 5, // client takes 5 SGD steps per weight update
    batchSize: 5, // batch size (if less than `examplesPerUpdate`)
    noiseStddev: 0.001 // client adds N(0, 0.001) noise to their updates
  }
})
```

Many of these hyperparameters matter a great deal for the efficiency and privacy of learning, but the correct settings depend greatly on the nature of the data, the size of the model being trained, and how consistently the data is distributed across clients. In the future, we hope to support automated (and dynamic) tuning of these hyperparameters.

### TODO

General:
- save and expose client-side performance metrics

Robustness:
- `median` and `trimmed-mean` aggregations (for [Byzantine-robustness](https://arxiv.org/abs/1803.01498))
- client authentication (e.g. gmail account + captcha)
- smoothing to limit individual clients' weight contributions (to prevent model from overfitting to most active clients and also create preconditions for Byzantine-robust learning if some clients are adversarial)
- create virtual server-side clients who minimize train loss
- discard client updates that increase server-side train loss (or subtract updates' projections onto the direction of increasing train loss)

Privacy:
- determine how to set hyperparameters such that each version of the model is differentially private to individual clients' updates (i.e. prevent sensitive information from leaking from client->server->client)
- consider implementing [secure aggregation](https://eprint.iacr.org/2017/281) (i.e. prevent sensitive information from leaking from client->server)
