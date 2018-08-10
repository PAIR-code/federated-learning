# Federated Learning in Tensorflow.js

## What is federated learning?

Federated Learning is a method to train a machine learning model in a decentralized manner.

It consists of a central coordinating server, and a number of clients. The basic process takes 5 steps:

1. A client connects to the server
2. The server sends the latest version of the model to the client
3. The client trains the model locally using some local data (for instance, images from a cameraphone)
4. The client uploads the new model to the server
5. The server waits for a number of clients to upload their models, and averages them. This averaged model is now broadcast to all the clients, and marked as the latest version.

Federated Learning lets the machine learning model be trained without the server ever seeing client data.

## What we'll be making

To do this, we'll need to implement a client and instantiate a server.

## Client


If we were starting from a pre-trained `tf.Model`, we can just pass its URL to the `federated.Client` constructor, as in the [client docs](src/client/README.md).

Since we're implementing our own model, instead we can use [`FederatedDynamicModel`](src/client/common.ts#L311). This class wraps a bag of variables, a loss function, and a predict function.

```js
import * as federated from 'federated-learning-client'

const SERVER_URL = 'http://localhost:3000'
const MODEL_PATH =
async main() {
  const client = new federated.Client(SERVER_URL, )
}

```

## Server

## Best Practices
