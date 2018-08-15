import * as tf from '@tensorflow/tfjs';
import * as federated from 'federated-learning-client';
import $ from 'jquery';
import model from './model';
import {Isotropic2DGaussian} from './gaussian';
import {MockServer} from 'federated-learning-mock-server';

window.tf = tf;
const verbose = true;

const mockServer = new MockServer(model, {
  verbose: verbose,
  serverHyperparams: {
    minUpdatesPerVersion: 5
  }
});

const RADIUS = 8;
const N_CLIENTS = 1;

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

const hyperparams = {
  learningRate: 1e-2,
  examplesPerUpdate: 10,
  epochs: 10,
  batchSize: 32,
  weightNoiseStddev: 0
};

const clients = range(N_CLIENTS).map(i => {
  return new federated.Client(mockServer.newClientSocket, model, {
    verbose: verbose,
    clientId: `client-${i}`,
    modelCompileConfig: {
      loss: 'binaryCrossentropy'
    },
    hyperparams: hyperparams,
    sendMetrics: true
  })
});
window.clients = clients;

const metricsByClient = {};
clients.forEach((c, i) => {
  metricsByClient[i] = [];
  c.onUpload(msg => metricsByClient[i].push(msg.metrics));
})

const means = [[-1,5], [6,4], [-4,-6]];
const stddevs = [4.0, 3.75, 6.25];
//const rates = [1000, 200, 400];

$('#updatesPerVersion').on('input', (e) => {
  const val = parseInt(e.target.value);
  mockServer.serverHyperparams.minUpdatesPerVersion = val;
  $('#updatesPerVersion + span').text(val);
}).val(mockServer.serverHyperparams.minUpdatesPerVersion).trigger('input');

$('#examplesPerUpdate').on('input', (e) => {
  const val = parseInt(e.target.value);
  hyperparams.examplesPerUpdate = val;
  $('#examplesPerUpdate + span').text(val);
  clients.forEach(c => c.hyperparams = hyperparams);
}).val(hyperparams.examplesPerUpdate).trigger('input');

$('#learningRate').on('input', (e) => {
  const val = 10**(parseFloat(e.target.value));
  hyperparams.learningRate = val;
  $('#learningRate + span').text(val);
  clients.forEach(c => c.hyperparams = hyperparams);
}).val(Math.log10(hyperparams.learningRate)).trigger('input');

$('#epochs').on('input', (e) => {
  const val = parseInt(e.target.value);
  hyperparams.epochs = val;
  $('#epochs + span').text(val);
  clients.forEach(c => c.hyperparams = hyperparams);
}).val(hyperparams.epochs).trigger('input');

$('#weightNoiseStddev').on('change', (e) => {
  const val = parseFloat(e.target.value);
  hyperparams.weightNoiseStddev = val;
  clients.forEach(c => c.hyperparams = hyperparams);
}).val(hyperparams.weightNoiseStddev);

const gaussians = range(N_CLIENTS).map(i =>
  new Isotropic2DGaussian(means[i][0], means[i][1], stddevs[i]));
window.gaussians = gaussians;

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
    Plotly.newPlot('clientLosses', clients.map((c, i) => {
      return {
        x: range(metricsByClient[i].length),
        y: metricsByClient[i].map(m => m[0]),
        type: 'scatter',
        mode: 'markers+lines',
        name: `Client ${i+1}`
      };
    }), {
      height: 400,
      width: 800,
      title: 'Client Losses',
      xaxis: { linecolor: 'black', mirror: true, title: 'Update Iteration' },
      yaxis: { linecolor: 'black', mirror: true, title: 'Training Loss' },
    }, {
      staticPlot: true
    });

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
      xaxis: { range: [-10, 10], title: 'Data x<sub>1</sub>' },
      yaxis: { range: [-10, 10], title: 'Data x<sub>2</sub>' },
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
      marker: { size: 10 },
      name: 'y=0'
    }, {
      x: truePoints.map(x => x[0]),
      y: truePoints.map(x => x[1]),
      type: 'scatter',
      mode: 'markers',
      marker: { size: 10 },
      name: 'y=1'
    }, {
      x: gaussians.map(g => g.x),
      y: gaussians.map(g => g.y),
      text: gaussians.map((g,i) => `<b>Client ${i+1}</b><br>${g.toString()}`),
      textposition: 'bottom',
      type: 'scatter',
      mode: 'markers+text',
      marker: { size: 15 },
      name: 'Clients'
    }], {
      height: 600,
      width: 617,
      title: 'Clients and Data',
      xaxis: { range: [-10, 10], linecolor: 'black', mirror: true, title: 'Data x<sub>1</sub>' },
      yaxis: { range: [-10, 10], linecolor: 'black', mirror: true, title: 'Data x<sub>2</sub>' },
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

function clamp(x, lower, upper) {
  return Math.max(lower, Math.min(x, upper));
}

let truePoints = [];
let falsePoints = [];

async function main() {
  await mockServer.setup();
  await Promise.all(clients.map(c => c.setup()));
  redraw();

  clients[0].onNewVersion(redraw);

  const callbacks = clients.map((client, i) => {
    return (done) => {
      const xs = [];
      const ys = [];
      const n = client.hyperparam('examplesPerUpdate');
      range(n).forEach(() => {
        const x = gaussians[i].sample();
        const y = 0 + ((x[0]*x[0] + x[1]*x[1]) < RADIUS*RADIUS);
        if (y) {
          truePoints.push(x);
        } else {
          falsePoints.push(x);
        }
        xs.push(x);
        ys.push([y]);
      });
      if (truePoints.length > N_CLIENTS * n) {
        truePoints = truePoints.slice(truePoints.length - N_CLIENTS * n);
      }
      if (falsePoints.length > N_CLIENTS * n) {
        falsePoints = falsePoints.slice(falsePoints.length - N_CLIENTS * n);
      }
      const xt = tf.tensor2d(xs);
      const yt = tf.tensor2d(ys);
      client.federatedUpdate(xt, yt).then(() => {
        tf.dispose([xt, yt]);
        done();
      });
    };
  });

  let running = false;
  const runners = callbacks.map((cb, i) => {
    return () => {
      if (running) {
        cb(runners[i]);
      }
    }
  });

  document.getElementById('startButton').addEventListener('click', () => {
    running = true;
    runners.forEach(r => r());
  });

  document.getElementById('pauseButton').addEventListener('click', () => {
    running = false;
  });

  $('#sd-range-inputs').html('');
  gaussians.forEach((g, i) => {
    const input = $('<input type="range" min="0.5" max="10.0" step="0.5"/>');
    const label = $(`<label>Client ${i+1} Stddev</label>`);
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
