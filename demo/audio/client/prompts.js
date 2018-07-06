const laterPrompt = 'If you\'re up for another, ' +
    'could you show me how to pronounce:';
const basicThanks = ['Thanks!', 'Gracias!', 'Much obliged!', 'Bravo!'];
let thanksVariants = [];
thanksVariants = thanksVariants.concat(basicThanks);
thanksVariants = thanksVariants.concat(basicThanks);
thanksVariants.push('Not bad!');
thanksVariants = thanksVariants.concat(basicThanks);
thanksVariants.push('You\'re getting good at this!');
thanksVariants = thanksVariants.concat(basicThanks);
thanksVariants = thanksVariants.concat(basicThanks);
thanksVariants.push('That was a lot, maybe you should take a break!');

let thanksIdx = 0;

export function firstPrompt() {
  return 'Would you be willing to help me?' +
    ' I\'d love it if you could show me how to pronounce the word:';
}

export function nextPrompt() {
  const prompt = thanksVariants[thanksIdx] + ' ' + laterPrompt;
  thanksIdx = (thanksIdx + 1) % thanksVariants.length;
  return prompt;
}
