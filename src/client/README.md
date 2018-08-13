# Tensorflow.js Federated Learning Client

This library sets up a simple websocket-based client for transmitting and receiving
Tensorflow.js model weights.

## Usage

### Basic

```js
import * as federated from 'tfjs-federated-learning-client';

const SERVER_URL = 'https://federated.learning.server';
const INIT_MODEL = 'https://initial.com/tf-model.json';
const client = new federated.Client(SERVER_URL, INIT_MODEL);

client.onNewVersion((model, oldVersion, newVersion) => {
  console.log(`updated model from ${oldVersion} to ${newVersion}`);
});

client.setup().then(() => {
  const yhat = client.predict(x); // make predictions!
  client.federatedUpdate(x, y); // train (and asynchronously update the server)!
});
```

After connecting to a server and downloading a model, the federated client
object exposes a `model` that allows you to call `predict`, `evaluate`, and
`federatedUpdate` on new data. Calling `federatedUpdate` will asynchronously
send updated weights to the server. If the server receives enough updates
from different clients, it will update its canonical copy of the model and
send a new version down, which will transparently replace the current one.

### Setting the Initial Model

During federated updates, the client and server will only communicate
changed weights, which for applications like transfer learning may be fairly
small in size compared to the full model. However, to begin the process, the
client must first obtain a copy of the full model.

This can be done in one of the following ways:

```js
new federated.Client(SERVER_URL, tfModel); // tf.Model instance
new federated.Client(SERVER_URL, 'https://initial.com/model.json'); // url pointing to one
new federated.Client(SERVER_URL, async () => { // async function returning one
  const model = await tf.loadModel('https://initial.com/model.json')
  return someCustomTransformationOfThe(model);
});
new federated.Client(SERVER_URL, federatedClientModel); // if you need custom behavior
```

You can pass a `FederatedClientModel` if you need custom behavior not supported by `tf.Model`s. See its [documentation](./models.ts) for the methods you will need to implement.

Currently, we do not support loading the model automatically from the server (e.g. just doing `federated.Client(SERVER_URL)`). If the model you want to load client-side is only hosted on your federated learning server, but is a `tf.Model`, you can implement this fairly simply by statically serving the model files. Here is an example of how to do this with `express`:

```js
// server
const app = express();
const dir = '/models';
app.use(express.static(dir));
const webServer = http.createServer(app);
const fedServer = new federated.Server(webServer, initModel, {
   modelDir: dir // newest model will be symlinked to ${dir}/latest
});
await fedServer.setup();
webServer.listen(80);

// client
const url = 'http://my.server';
const fedClient = new federated.Client(url, `${url}/latest/model.json`);
await fedClient.setup();
```

The reason we recommend hosting the model elsewhere is to reduce unnecessary load on the server, which is not optimized to serve large static files.

### Federated Learning

Simply call:

```js
await client.federatedUpdate(inputTensor, targetTensor);
```

The `federatedUpdate` method will:
- verify `inputTensor` and `targetTensor` have the correct shapes (can pass inputs either with or without the batch dimension)
- check if it has enough examples to train (this is a hyperparameter sent by the server, which can be inspected on the client using `client.numExamplesPerUpdate()`)
- if there are enough examples, it will fit the model, send updated weights to the server, and then revert back to its previous weights
- if there aren't, it will store the examples in memory (copying the tensors) and immediately resolve.

### Setting the Loss Function

For `tf.Model`-based federated learning, we assume by default that we are minimizing
the `categoricalCrossEntropy` when users call `client.fit(x, y)`. If this assumption is incorrect, you can do the following:

```js
federated.Client(SERVER_URL, 'https://initial.com/model.json', {
  modelCompileConfig: {
    loss: 'meanSquaredError'
  }
});
```

This dictionary will be passed to [`tf.Model.compile`](https://js.tensorflow.org/api/latest/#tf.Model.compile), so you can also pass custom `metrics`.
However, we will always set the `optimizer` to [`tf.SGDOptimizer`](https://js.tensorflow.org/api/latest/#train.sgd) to properly adopt a learning rate based on hyperparameters sent by the server. If you need more flexibility than this, you can define
a custom [`FederatedClientModel`](./models.ts).

### Getting Stats

For printing and debugging purposes, you can get stats about the client using:
```js
client.inputShape     // shape of model inputs (without batch dimension)
client.outputShape    // shape of model outputs (without batch dimension)
client.evaluate(x, y) // metrics for the current model on the data
client.predict(x)     // predictions of current model on the data
client.modelVersion() // identifier of the current model
client.numUpdates() // how many updates client has contributed
client.numVersions() // how many versions client has downloaded
client.numExamples() // how many examples currently stored in client's buffer
client.numExamplesPerUpdate() // how many examples used in each update
client.numExamplesRemaining() // how many more examples needed before updating
```

You can also listen for changes in the model version by doing:
```js
client.onNewVersion((model, oldVersion, newVersion) => {
  console.log(`you've seen ${client.numVersions()} versions now!`);
  $('#model-version').text(`Model Version #${newVersion}`);
});
```

### TODO

- Support initialization of clients without initial models (i.e. autoloading them from the server)
- Support customizable client authentication (e.g. google oauth, captchas, etc)
- Collect a local dataset of examples we store even after they're used in `federatedUpdate`. Change the behavior of `federatedUpdate` to compute weight updates relative to the original model but then retrain on the full local dataset -- so clients can see the benefits of training even before the server model adapts.
