# Tensorflow.js Federated Learning Server

This library sets up a simple websocket-based server for transmitting and receiving
TensorflowJS model weights.

## Usage

### Basic

```js
import * as http from 'http';
import * as federated from 'tfjs-federated-learning-server';

const INIT_MODEL = 'file:///initial/model.json';
const httpServer = http.createServer();
const fedServer = new federated.Server(httpServer, INIT_MODEL);

fedServer.setup().then(() => {
  httpServer.listen(8080);
});
```

### Setting the Initial Model

```js
new federated.Server(httpServer, tfModel); // Initialize a federated server from an in-memory tf.Model
new federated.Server(httpServer, 'https://remote.server/tf-model.json'); // or from a URL pointing to one
new federated.Server(httpServer, 'file:///my/local/file/tf-model.json'); // (which can be a file URL in Node)
new federated.Server(httpServer, async () => { // or from an asynchrous function returning one
  const model = await tf.loadModel('file:///transfer/learning/model.json');
  model.layers[0].trainable = false;
  return model;
});
new federated.Server(httpServer, federatedServerModel); // if you need fully custom behavior; see below
```

The simplest way to set up a `federated.Server` is to pass a [`tf.Model`](https://js.tensorflow.org/api/latest/#class:Model). However, you can also pass a string that will be delegated to [`tf.loadModel`](https://js.tensorflow.org/api/latest/#loadModel) (both `https?://` and `file://` URLs should work), or an asynchronous function that will return a `tf.Model`. The final option is to define your own `FederatedServerModel`, which has to implement various saving and loading methods. See its [documentation](./models.ts) for more details.

Note that by default, different `tf.Model` versions will be saved as files in subfolders of `${process.cwd()}/saved-models/`. If you would like to change this directory, you can pass a `modelDir` configuration parameter, e.g. `federated.Server(httpServer, model, { modelDir: '/mnt/my-vfs' })`.

### Setting Hyperparameters

```js
new federated.Server(httpServer, model, {
  // These are true server parameters
  serverHyperparams: {
    aggregation: 'mean',      // how to merge weights (only mean supported now)
    minUpdatesPerVersion: 20, // server merges every 20 client weight updates
  },
  // These get broadcast to clients
  clientHyperparams: {
    learningRate: 0.01,    // client takes SGD steps of size 0.01
    epochs: 5,             // client takes 5 SGD steps per weight update
    examplesPerUpdate: 10, // client computes weight updates every 10 examples
    batchSize: 5,          // client subdivides `examplesPerUpdate` into batches
    noiseStddev: 0.001     // client adds N(0, 0.001) noise to their updates
  },
  verbose: false,           // whether to print debugging/timing information
  modelDir: '/mnt/my-vfs',  // server stores tf.Model-specific versions here
  modelCompileConfig: {     // tf.Model-specific compile config
    loss: 'categoricalCrossEntropy',
    metrics: ['accuracy']
  }
})
```

Many of these hyperparameters matter a great deal for the efficiency and privacy of learning, but the correct settings depend greatly on the nature of the data, the size of the model being trained, and how consistently the data is distributed across clients. In the future, we hope to support automated (and dynamic) tuning of these hyperparameters.

### Listening to Events

You can add an event listener that fires each time a client uploads a new set of weights (and optionally, self-reported metrics of how well the model performed on the examples used in training):

```js
fedServer.on('upload', message => {
  console.log(message.model.version); // version of the model
  console.log(message.model.vars); // serialized model variables
  console.log(message.clientId); // self-reported and usually random client ID
  console.log(message.metrics); // array of performance metrics for the update; only sent for clients configured to `sendMetrics`
});
```

You can also listen for whenever the server computes a new version of the model:

```js
fedServer.on('new-version', (oldVersion, newVersion) => {
  console.log(`updated model from ${oldVersion} to ${newVersion}`);
});
```

### TODO

Robustness:
- `median` and `trimmed-mean` aggregations (for [Byzantine-robustness](https://arxiv.org/abs/1803.01498))
- client authentication (e.g. google oauth, captchas)
- smoothing to limit individual clients' weight contributions (to prevent model from overfitting to most active clients and also create preconditions for Byzantine-robust learning if some clients are adversarial)
- create virtual server-side clients who minimize train loss
- discard client updates that increase server-side train loss (or subtract updates' projections onto the direction of increasing train loss)

Privacy:
- determine how to set hyperparameters such that each version of the model is differentially private to individual clients' updates (i.e. prevent sensitive information from leaking from client->server->client)
- consider implementing [secure aggregation](https://eprint.iacr.org/2017/281) (i.e. prevent sensitive information from leaking from client->server)
