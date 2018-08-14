import * as tf from '@tensorflow/tfjs';
import * as federated from 'federated-learning-client';
import * as poissonProcess from 'poisson-process';
import $ from 'jquery';
import model from './model';

const SERVER_URL = 'http://localhost:3000';
const RADIUS = 8;
const N_CLIENTS = 3;

function range(n) {
  return Array.from(new Array(n).keys());
}

const x1s = [];
const x2s = [];
for (let x = -10; x <= 10; x += 0.5) {
  x1s.push(x);
  x2s.push(x);
}
const xs = [];
x1s.forEach(x1 => {
  x2s.forEach(x2 => {
    xs.push([x1, x2]);
  })
})
const xTensor = tf.tensor2d(xs);

const clients = range(N_CLIENTS).map(i =>
  new federated.Client(SERVER_URL, model, {
    verbose: true,
    clientId: `client-${i}`,
    modelCompileConfig: {
      loss: 'binaryCrossentropy'
    }
  })
);
window.clients = clients;

const means = [[-1,5], [6,4], [-4,-6]];
const stddevs = [4.0, 3.75, 6.25];
const rates = [1000, 200, 400];

function redraw() {
  const zArray = tf.tidy(() => clients[0].predict(xTensor).dataSync());
  const zs = [];
  let vLim = 0;
  for (let i = 0; i < x1s.length; i++) {
    const row = [];
    for (let j = 0; j < x2s.length; j++) {
      const el = zArray[i * x1s.length + j];
      vLim = Math.max(vLim, el);
      vLim = Math.max(vLim, -el);
      row.push(el);
    }
    zs.push(row);
  }

  function resetPlot() {
    Plotly.newPlot('canvasContainer', [{
      x: x1s,
      y: x2s,
      z: zs,
      type: 'contour',
      colorscale: 'RdBu',
      contours: {
        start: 0,
        end: 1,
        size: 0.01
      }
    }, {
      x: x1s,
      y: x2s,
      z: zs,
      type: 'contour',
      contours: {
        coloring: 'lines',
        start: 0.499999,
        end: 0.500001,
      },
      showscale: false,
      line: {width: 3}
    }], {
      height: 600,
      width: 617,
      title: 'True vs. Learned Decision Boundary',
      xaxis: { range: [-10, 10] },
      yaxis: { range: [-10, 10] },
      shapes: [{
        type: 'circle',
        xref: 'x',
        yref: 'y',
        x0: -RADIUS, x1: RADIUS, y0: -RADIUS, y1: RADIUS,
        line: {
          color: 'rgba(0,0,0,1)'
        }
      }]
    }, {
      staticPlot: true
    });

    Plotly.newPlot('pointsContainer', [{
      x: falsePoints.map(x => x[0]),
      y: falsePoints.map(x => x[1]),
      type: 'scatter',
      mode: 'markers',
      marker: { size: 10 }
    }, {
      x: truePoints.map(x => x[0]),
      y: truePoints.map(x => x[1]),
      type: 'scatter',
      mode: 'markers',
      marker: { size: 15 }
    }, {
      x: gaussians.map(g => g.x),
      y: gaussians.map(g => g.y),
      text: gaussians.map((g,i) => `<b>Client ${i+1}</b><br>${g.toString()}`),
      textposition: 'bottom',
      type: 'scatter',
      mode: 'markers+text',
      marker: { size: 15 }
    }], {
      height: 600,
      width: 617,
      title: 'Clients and Data',
      xaxis: { range: [-10, 10], linecolor: 'black', mirror: true },
      yaxis: { range: [-10, 10], linecolor: 'black', mirror: true },
      shapes: range(N_CLIENTS).map(i => {
        return {
          type: 'circle',
          xref: 'x',
          yref: 'y',
          x0: gaussians[i].x - gaussians[i].sd,
          x1: gaussians[i].x + gaussians[i].sd,
          y0: gaussians[i].y - gaussians[i].sd,
          y1: gaussians[i].y + gaussians[i].sd,
          line: {
            color: 'rgba(50,171,96,0.5)'
          }
        }
      })
    }, {
      staticPlot: true
    });

    Plotly.newPlot('clientKLs', [{
      x: gaussians.map((g, i) => `Client ${i+1}`),
      y: gaussians.map((g, i) => `Client ${i+1}`),
      z: gaussians.map(g1 => gaussians.map(g2 => g1.kl(g2))),
      type: 'heatmap',
      colorscale: 'Electric'
    }], {
      height: 400,
      width: 400,
      title: 'KL Divergence Between Client Distributions'
    }, {
      staticPlot: true
    })
  }

  resetPlot();

  function startDragBehavior() {
      var d3 = Plotly.d3;
      var drag = d3.behavior.drag();
      let xOffset;
      let yOffset;
      drag.on("dragstart", (opts) => {
        const container = document.getElementById('pointsContainer');
        const xmouse = d3.event.sourceEvent.clientX;
        const ymouse = d3.event.sourceEvent.clientY;
        var xaxis = container._fullLayout.xaxis;
        var yaxis = container._fullLayout.yaxis;
        const x = xaxis.p2l(xmouse);
        const y = yaxis.p2l(ymouse);
        xOffset = x - opts.x;
        yOffset = y - opts.y;
      });
      drag.on("drag", (opts) => {
        const container = document.getElementById('pointsContainer');
        const xmouse = d3.event.sourceEvent.clientX;
        const ymouse = d3.event.sourceEvent.clientY;
        var xaxis = container._fullLayout.xaxis;
        var yaxis = container._fullLayout.yaxis;
        const x = clamp(xaxis.p2l(xmouse) - xOffset, -10, 10);
        const y = clamp(yaxis.p2l(ymouse) - yOffset, -10, 10);
        gaussians[opts.i].x = x;
        gaussians[opts.i].y = y;
        resetPlot();
      });
      drag.on("dragend", () => {
        d3.selectAll(".scatterlayer .trace:last-of-type .points path").call(drag);
      });
      d3.selectAll(".scatterlayer .trace:last-of-type .points path").call(drag);
  }

  startDragBehavior();
}

function randomNormal(mu, sd) {
  return tf.tidy(() => tf.randomNormal([1], mu, sd).dataSync()[0]);
}

function clamp(x, lower, upper) {
  return Math.max(lower, Math.min(x, upper));
}

function round(x, to) {
  return Math.round(x * 10**to) / 10**to;
}


class Isotropic2DGaussian {
  constructor(x, y, sd) {
    this.x = x;
    this.y = y;
    this.sd = sd;
  }

  sample() {
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

const gaussians = range(N_CLIENTS).map(i =>
  new Isotropic2DGaussian(means[i][0], means[i][1], stddevs[i]));

let truePoints = [];
let falsePoints = [];

async function main() {
  await Promise.all(clients.map(c => c.setup()));
  redraw();

  clients[0].onNewVersion(redraw);

  const callbacks = clients.map((client, i) => {
    return () => {
      const x = gaussians[i].sample();
      const y = 0 + ((x[0]*x[0] + x[1]*x[1]) < RADIUS*RADIUS);
      if (y) {
        truePoints.push(x);
        if (truePoints.length > 100) {
          truePoints = truePoints.slice(1);
        }
      } else {
        falsePoints.push(x);
        if (falsePoints.length > 100) {
          falsePoints = falsePoints.slice(1);
        }
      }
      client.federatedUpdate(tf.tensor2d([x]), tf.tensor1d([y]));
    };
  });
  let intervals;

  document.getElementById('startButton').addEventListener('click', () => {
    intervals = callbacks.map(c => setInterval(c, 100));
  });

  document.getElementById('pauseButton').addEventListener('click', () => {
    intervals.map(int => clearInterval(int));
  });

  $('#sd-range-inputs').html('');
  gaussians.forEach((g, i) => {
    const input = $('<input type="range" min="0.5" max="10.0" step="0.5"/>');
    const label = $(`<label>Client ${i+1}</label>`);
    label.append(input);
    $('#sd-range-inputs').append(label);
    input.on('input', () => {
      g.sd = parseFloat(input.val());
      redraw();
    });
    input.val(g.sd);
 });
}

main();
