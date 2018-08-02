Federated learning server using TensorflowJS

## Usage

### Basic

```js
import * as express from 'express';
import * as http from 'http';
import * as federated from 'tfjs-federated-learning-server';

const expressApp = express();
const httpServer = http.createServer(expressApp);
const server = federated.Server(httpServer, 'https://initial.com/model.json');

server.setup().then(() => {
  httpServer.listen(PORT);
})
```

### Setting the Initial Model

```js
// Examples
federated.Server(httpServer, tfModelInstance);
federated.Server(httpServer, 'https://remote.server/tf-model.json');
federated.Server(httpServer, 'file:///my/local/file/tf-model.json');
federated.Server(httpServer, async () => {
  const model = await tf.loadModel('tf-model.json');
  model.layers[0].trainable = false;
  return model;
});
federated.Server(httpServer, federatedModelInstance); // see below
```

The simplest way to set up a federated learning server is to pass a [`tf.Model`](https://js.tensorflow.org/api/0.12.0/#class:Model). However, you can also pass a string that will be delegated to [`tf.loadModel`](https://js.tensorflow.org/api/0.12.0/#loadModel) (both `https?://` and `file://` URLs should work), or an asynchronous function that will return a `tf.Model`. The final option is to define your own `FederatedServerModel`, which has to implement various saving and loading methods. See its [documentation](#TODO) for more details.

Note that by default, different `tf.Model` versions will be saved as files in subfolders of `${__dirname}/federated-server-models/`. If you would like to change this directory, you can pass a `modelDir` configuration parameter, e.g. `federated.Server(httpServer, model, { modelDir: '/mnt/my-vfs' })`.

### Setting Hyperparameters

```js
// Examples
federated.Server(httpServer, 'model.json', {
  // These hyperparams only affect the server
  serverHyperparams: {
    updatesPerVersion: 20, // server merges every 20 client weight updates
    weightAggregator: 'mean' // how to merge weights (only mean supported now)
  },
  // These get broadcast to clients
  clientHyperparams: {
    examplesPerUpdate: 10, // client computes weight updates every 10 examples
    learningRate: 0.01, // client takes SGD steps of size 0.01
    epochs: 5, // client takes 5 SGD steps per weight update
    batchSize: 5, // batch size (if less than `examplesPerUpdate`)
    noiseStddev: 0.001 // client adds noise ~ N(0, 0.001) to their updates
  }
})
```

Many of these hyperparameters matter a great deal for the efficiency and privacy of learning, but the correct settings depend greatly on the nature of the data, the size of the model being trained, and how consistently the data is distributed across clients. In the future, we hope to support dynamic automated tuning of these hyperparameters.

