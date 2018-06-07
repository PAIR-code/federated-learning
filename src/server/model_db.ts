import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as path from 'path';

const DEFAULT_MIN_UPDATES = 10;

function getLatestId(dir: string): string {
  return fs.readdirSync(dir).reduce((acc, val) => {
    if (val.endsWith('.json') && val.slice(0, -5) > acc) {
      return val.slice(0, -5);
    } else {
      return acc;
    }
  }, '0');
}

function generateNewId() {
  return new Date().getTime().toString();
}

function readJSON(path: string): any {
  return JSON.parse(fs.readFileSync(path).toString());
}

export class ModelDB {
  dataDir: string;
  modelId: string;
  updating: boolean;
  minUpdates: number;

  constructor(dataDir: string, currentModel?: string, minUpdates?: number) {
    this.dataDir = dataDir;
    this.modelId = currentModel || getLatestId(dataDir);
    this.updating = false;
    this.minUpdates = minUpdates || DEFAULT_MIN_UPDATES;
  }

  updateFiles(): string[] {
    return fs.readdirSync(path.join(this.dataDir, this.modelId));
  }

  currentVars(): tf.Tensor[] {
    const file = path.join(this.dataDir, this.modelId + '.json');
    const json = readJSON(file);
    return json['vars'].map((v: {values: number[], shape: number[]}) => {
      return tf.tensor(v.values, v.shape);
    });
  }

  possiblyUpdate() {
    if (this.updateFiles().length < this.minUpdates || this.updating) {
      return;
    }
    this.updating = true;
    this.update();
    this.updating = false;
  }

  update() {
    const updatedVars = this.currentVars();
    const updateFiles = this.updateFiles();
    const updatesJSON = updateFiles.map(readJSON);

    // Compute total number of examples for normalization
    let totalNumExamples = 0;
    updatesJSON.forEach((obj) => {
      totalNumExamples += obj['num_examples'];
    });
    const n = tf.scalar(totalNumExamples);

    // Apply normalized updates
    updatesJSON.forEach((u) => {
      const nk = tf.scalar(u['num_examples']);
      const frac = nk.div(n);
      u['vars'].forEach((v: {values: number[], shape: number[]}, i: number) => {
        const update = tf.tensor(v.values, v.shape).mul(frac);
        updatedVars[i] = updatedVars[i].add(update);
      });
    });

    // Save results and update key
    const newModelId = generateNewId();
    const newModelDir = path.join(this.dataDir, newModelId);
    const newModelPath = path.join(this.dataDir, newModelId + '.json');
    const newModelJSON = JSON.stringify({
      'vars': updatedVars.map((v) => {
        return {'values': v.dataSync(), 'shape': v.shape};
      })
    });
    fs.writeFileSync(newModelPath, newModelJSON);
    fs.mkdirSync(newModelDir);
    this.modelId = newModelId;
  }
}
