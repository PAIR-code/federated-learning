# Federated Emoji Hunt Demo

In this demo, we build off the idea of the [emoji scavenger hunt](https://emojiscavengerhunt.withgoogle.com/). The goal of the scavenger hunt is to find a real-world object that our image model recognizes as (corresponding to) a particular emoji. However, the model we're using isn't perfect, so in case it fails to recognize an object, you can click "I'm looking at it," and the model will update itself using our federated learning library.

## Running the Demo

To start the server:
```
cd demo/emoji_hunt/server
yarn dev
```

To start the client:
```
cd demo/emoji_hunt/client
yarn watch
```
