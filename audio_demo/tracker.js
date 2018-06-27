export default function Tracker(waitingPeriod, refractoryPeriod) {
  this.waitingPeriod = waitingPeriod;
  this.refractoryPeriod = refractoryPeriod;
  this.counter = 0;
  this.lastTriggerCounter = -1;

  // 0 - resting, waiting for trigger.
  // 1 - triggered, waiting until waitingPeriod passes.
  // 2 - firing
  // 3 - fired, in refractoryPeriod.
  this.state = 0;

  this.tick = (trigger) => {
    if (this.state == 0) {
      if (trigger) {
        this.lastTriggerCounter = this.counter;
        this.state = 1;
      }
    } else if (this.state === 1) {
      if (this.counter - this.lastTriggerCounter === this.waitingPeriod) {
        this.state = 2;
      }
    } else if (this.state === 2) {
      this.state = 3;
    } else if (this.state === 3) {
      // In refractory period.
      if (this.counter - this.lastTriggerCounter ===
          this.waitingPeriod + 1 + this.refractoryPeriod) {
        this.state = 0;
      }
    }
    this.counter++;
  }

  this.shouldFire = () => this.state === 2;

  this.isResting = () => this.state === 0;

  this.reset = () => {
    this.state = 0;
    this.counter = 0;
    this.lastTriggerCounter = -1;
  }
}
