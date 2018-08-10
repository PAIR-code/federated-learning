# Tensorflow.js Federated Learning Client

This library sets up a simple websocket-based client for transmitting and receiving
Tensorflow.js model weights.

## Usage

### Basic

```js
import * as federated from 'tfjs-federated-learning-client';

const SERVER_URL = 'https://federated.learning.server';
const INIT_MODEL = 'https://initial.com/tf-model.json';
const client = federated.Client(SERVER_URL, INIT_MODEL);

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

After connecting to a server and downloading a model, the federated client
object exposes a `model` that allows you to `fit`, `predict`, and `evaluate`
on new data. Calling `fit` will asynchronously send updated weights to the
server. If the server receives enough updates from different clients, it will
update its canonical copy of the model and send a new version down, which
will transparently replace the current one.

### Setting the Initial Model

During federated training, the client and server will only communicate
updated weights, which for applications like transfer learning may be fairly
small in size relative to the full model. However, to begin the process, the
client must first obtain a copy of the full model.

We currently do not yet support loading the full model from the server, though
we plan to support this in the future. However, to avoid unnecessary load on the
server, it's actually generally more efficient
to load it from elsewhere, like this:

```js
federated.Client(SERVER_URL, 'https://initial.com/model.json')
federated.Client(SERVER_URL, tfModel)
federated.Client(SERVER_URL, async () => {
  const model = await tf.loadModel('https://initial.com/model.json')
  return someCustomTransformationOfThe(model);
})
federated.Client(SERVER_URL, federatedClientModel)
```

You can also pass a `FederatedClientModel` if you need custom behavior not supported by `tf.Model`s. See its [documentation](#TODO) for the methods you will need to implement.

### Setting the Loss Function

For `tf.Model`-based federated learning, we assume by default that we are minimizing
the `categoricalCrossEntropy` when users call `client.fit(x, y)`. If this assumption is incorrect, you can do the following:

```js
federated.Client(SERVER_URL, 'https://initial.com/model.json', {
  modelCompileConfig: {
    loss: 'meanSquaredError'
  }
})
```

This dictionary will be passed to [`tf.Model.compile`](https://js.tensorflow.org/api/latest/#tf.Model.compile), so you can also pass custom `metrics`.
However, we will always set the `optimizer` to [`tf.SGDOptimizer`](https://js.tensorflow.org/api/latest/#train.sgd) to properly adopt a learning rate based on hyperparameters sent by the server. If you need more flexibility than this, you can define
a custom [`FederatedClientModel`](#TODO).

### Getting Stats

For printing and debugging purposes, you can get stats about the client using:
```js
client.modelVersion() // identifier of the current model
client.numUpdates() // how many updates client has contributed
client.numVersions() // how many versions client has downloaded
client.numExamples() // how many examples client has trained on
client.numExamplesUntilUpdate() // how many more examples needed before updating
```

You can also listen for changes in the model version by doing:
```js
client.onNewVersion((model, oldVersion, newVersion) => {
  console.log(`you've seen ${client.numVersions()} versions now!`);
  $('#model-version').text(`Model Version #${newVersion}`);
});
```

### TODO

- client authentication (e.g. gmail account + captcha)
