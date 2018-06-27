import * as tf from '@tensorflow/tfjs';
import {Scalar, Tensor} from '@tensorflow/tfjs';
import * as path from 'path';

import {FederatedModel, ModelDict} from '../types';

import {setup} from './server';

const dataDir = path.resolve(__dirname + '/../../data');

const audioTransferLearningModelURL =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/tfjs-speech-command-model-14w/model.json';

class AudioTransferLearningModel implements FederatedModel {
  async setup(): Promise<ModelDict> {
    const model = await tf.loadModel(audioTransferLearningModelURL);

    for (let i = 0; i < 9; ++i) {
      model.layers[i].trainable = false;  // freeze conv layers
    }

    const loss = (inputs: Tensor, labels: Tensor) => {
      const logits = model.predict(inputs) as Tensor;
      const losses = tf.losses.softmaxCrossEntropy(logits, labels);
      return losses.mean() as Scalar;
    };

    return {predict: model.predict, vars: model.trainableWeights, loss};
  }
}

const model = new AudioTransferLearningModel();
setup(model, dataDir);
