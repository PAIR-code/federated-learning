import p5 from 'p5';

const container = document.querySelector('#canvasContainer');

const RESOLUTION = 25;
const CANVAS_SIZE = 400;
const CIRCLE_DIAM = 2 * (CANVAS_SIZE / 3);

const DOT_SIZE = CANVAS_SIZE / 20;

const CELL_SIZE = CANVAS_SIZE / RESOLUTION;

const clickLocations = [];
export const onClick = []

let decisionBoundary = new Float32Array(RESOLUTION * RESOLUTION).fill(0.0);

const LABEL_COLOURS = {
  0: [255, 0, 0],
  1: [0, 0, 255]
}

function p5Sketch(p) {

  p.setup = function() {
    const canvas = p.createCanvas(CANVAS_SIZE, CANVAS_SIZE);
    container.appendChild(canvas.elt);
  }

  p.draw = function() {
    p.background('white');
    p.noStroke();

    for(let i = 0; i < RESOLUTION; i++) {
      for(let j = 0; j < RESOLUTION; j++) {
        const v = decisionBoundary[i * RESOLUTION + j]
        if(v < 0.0) {
          p.fill(...LABEL_COLOURS[0], -v * 255);
        } else {
          p.fill(...LABEL_COLOURS[1], v * 255);
        }
        p.rect(i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      }
    }

    for(const clickLocation of clickLocations) {
      const {label, x, y} = clickLocation;
      p.fill(LABEL_COLOURS[label]);
      p.ellipse(x, y, DOT_SIZE, DOT_SIZE);
    }

    p.noFill();
    p.stroke('black');

    p.ellipse(p.width / 2, p.height / 2, CIRCLE_DIAM, CIRCLE_DIAM);
  }

  p.mousePressed = function() {
    const dist = p.dist(p.mouseX, p.mouseY, p.width / 2, p.height / 2);
    let dataPoint;
    console.log(dist, CIRCLE_DIAM);
    if(2 * dist < CIRCLE_DIAM) {
      dataPoint = {x: p.mouseX, y: p.mouseY, label: 0};
    } else {
      dataPoint = {x: p.mouseX, y: p.mouseY, label: 1};
    }
    clickLocations.push(dataPoint);
    onClick.forEach(callback => callback(dataPoint));
  }
}

export function setupUI() {
  return new p5(p5Sketch)
}
