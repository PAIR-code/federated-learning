import * as tf from '@tensorflow/tfjs';

function randomNormal(mu, sd) {
  return tf.tidy(() => tf.randomNormal([1], mu, sd).dataSync()[0]);
}

function round(x, to) {
  return Math.round(x * 10**to) / 10**to;
}

export class Isotropic2DGaussian {
  constructor(x, y, sd) {
    this.x = x;
    this.y = y;
    this.sd = sd;
  }

  sample(n) {
    const x = randomNormal(this.x, this.sd);
    const y = randomNormal(this.y, this.sd);
    return [x, y];
  }

  kl(other) {
    return 0.5 * (
      2 * (other.sd / this.sd) +
      (Math.pow(this.x - other.x, 2) + Math.pow(this.y - other.y, 2)) / this.sd +
      (-2) +
      Math.log(Math.pow(this.sd, 2) / Math.pow(other.sd, 2))
    );
  }

  toString() {
    return `<i>N</i>([${round(this.x, 2)},${round(this.y, 2)}], ${this.sd}<sup>2</sup><i>I</i>)`;
  }
}
