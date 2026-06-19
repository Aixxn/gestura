interface FrameResponse {
  status?: string;
  word?: unknown;
}

export const appendDetectedWord = (
  currentGloss: string,
  response: FrameResponse
) => {
  if (response.status !== 'word_detected' || typeof response.word !== 'string') {
    return currentGloss;
  }

  const word = response.word.trim();
  if (!word) {
    return currentGloss;
  }

  const words = currentGloss.trim() ? currentGloss.trim().split(/\s+/) : [];
  if (words[words.length - 1] === word) {
    return words.join(' ');
  }

  return [...words, word].join(' ');
};
