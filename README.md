# Federated Learning in TensorFlow.js

## *This is not an official federated learning framework for TensorFlow. This is an experimental library for TensorFlow.js that is currently ***unmaintained***. If you would like to use an official federated learning library, check out [tensorflow/federated](https://github.com/tensorflow/federated).*

This is the parent repository for an (experimental and demonstration-only)
implementation of [Federated
Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
in [Tensorflow.js](https://js.tensorflow.org/). Federated Learning is a
method for training machine learning models in a distributed fashion.
Although it involves a central server, that server never needs to see any
data or even compute a gradient. Instead, _clients_ perform all of the
inference and training locally (which they already do in Tensorflow.js), and
just periodically send the server updated weights (rather than data). The
server's only job is to aggregate and redistribute them, which means it can
be extremely lightweight!

## Basic Usage

On the server (NodeJS) side:

```js
import * as http from 'http';
import * as federated from 'federated-learning-server';

const INIT_MODEL = 'file:///initial/model.json';
const webServer = http.createServer(); // can also use https
const fedServer = new federated.Server(webServer, INIT_MODEL);

fedServer.setup().then(() => {
  webServer.listen(80);
});
```

On the client (browser) side:

```js
import * as federated from 'federated-learning-client';

const INIT_MODEL = 'http://my.initial/model.json';
const SERVER_URL = 'http://federated.learning.server'; // URL of server above
const client = new federated.Client(SERVER_URL, INIT_MODEL);

client.setup().then(() => {
  const yhat = client.predict(x); // make predictions!
  client.federatedUpdate(x, y);   // train and update the server!
});
```

## Documentation and Examples

See the [server](./src/server) and [client](./src/client)
READMEs for documentation, and the [emoji](./demo/emoji_hunt) or
[Hogwarts](./demo/audio) demos for more fully fleshed out examples.
