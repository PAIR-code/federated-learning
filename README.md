Federated learning experiment using TensorFlow.js

## Basic Usage

On the server side:

```js
import * as express from 'express';
import * as http from 'http';
import * as federated from 'tfjs-federated-server';

const app = express();
const server = http.createServer(app);

const api = federated.ServerAPI(
  server,
  'https://www.example.com/initial-model.json'
);

api.setup().then(() => {
  server.listen(PORT);
})
```

On the client side:

```js
import * as federated from 'tfjs-federated-client';

const api = federated.ClientAPI(
  SERVER_URL,
  'https://www.example.com/initial-model.json'
);

api.setup().then(() => {
  console.log(api.model.predict(INPUTS));
  api.federatedFit(INPUTS, LABELS);
});
```

## Advanced

See [server docs]()

