import 'mocha';
import { expect } from 'chai';
import { normalizeFrames } from '@src/routes/api';

describe('normalizeFrames()', () => {
  const FD = 1663;

  const makeFrames = (count: number, dim: number = FD): number[][] =>
    Array.from({ length: count }, (_, i) => Array.from({ length: dim }, (_, j) => i + j * 0.001));

  describe('edge cases', () => {
    it('returns [] when given an empty array', () => {
      expect(normalizeFrames([], 80)).to.deep.equal([]);
    });

    it('returns the same array when lengths already match', () => {
      const frames = makeFrames(80);
      const result = normalizeFrames(frames, 80);
      expect(result).to.equal(frames);
      expect(result).to.deep.equal(frames);
    });

    it('handles targetFrames of 1 (extreme downsampling)', () => {
      const frames = makeFrames(100);
      const result = normalizeFrames(frames, 1);
      expect(result).to.have.length(1);
      expect(result[0]).to.deep.equal(frames[0]);
    });

    it('handles a single frame padded to target size', () => {
      const frames = makeFrames(1);
      const result = normalizeFrames(frames, 80);
      expect(result).to.have.length(80);
      result.forEach(frame => {
        expect(frame).to.deep.equal(frames[0]);
      });
    });
  });

  describe('downsampling (n > targetFrames)', () => {
    it('preserves first and last frames', () => {
      const frames = makeFrames(100);
      const result = normalizeFrames(frames, 80);
      expect(result[0]).to.deep.equal(frames[0]);
      expect(result[result.length - 1]).to.deep.equal(frames[frames.length - 1]);
    });

    it('produces exactly targetFrames frames', () => {
      expect(normalizeFrames(makeFrames(200), 80)).to.have.length(80);
      expect(normalizeFrames(makeFrames(120), 80)).to.have.length(80);
      expect(normalizeFrames(makeFrames(81), 80)).to.have.length(80);
    });

    it('selects uniformly spaced indices', () => {
      const frames = makeFrames(80);
      const result = normalizeFrames(frames, 40);
      const expectedIndices = Array.from({ length: 40 }, (_, i) =>
        Math.round((79 * i) / 39)
      );
      expectedIndices.forEach((idx, i) => {
        expect(result[i]).to.deep.equal(frames[idx]);
      });
    });
  });

  describe('padding (n < targetFrames)', () => {
    it('repeats the last frame for padding', () => {
      const frames = makeFrames(50);
      const result = normalizeFrames(frames, 80);
      expect(result).to.have.length(80);
      for (let i = 0; i < 50; i++) {
        expect(result[i]).to.deep.equal(frames[i]);
      }
      const last = frames[49];
      for (let i = 50; i < 80; i++) {
        expect(result[i]).to.deep.equal(last);
      }
    });

    it('produces exactly targetFrames frames', () => {
      expect(normalizeFrames(makeFrames(10), 80)).to.have.length(80);
      expect(normalizeFrames(makeFrames(40), 80)).to.have.length(80);
      expect(normalizeFrames(makeFrames(79), 80)).to.have.length(80);
    });
  });

  describe('feature dimension integrity', () => {
    it('preserves inner array lengths', () => {
      const result = normalizeFrames(makeFrames(100, FD), 80);
      result.forEach(frame => {
        expect(frame).to.have.length(FD);
      });
    });

    it('preserves inner array lengths when padding', () => {
      const result = normalizeFrames(makeFrames(50, FD), 80);
      result.forEach(frame => {
        expect(frame).to.have.length(FD);
      });
    });

    it('preserves feature values correctly after padding (last frame copy)', () => {
      const frames = makeFrames(3, 4);
      const result = normalizeFrames(frames, 6);
      expect(result).to.have.length(6);
      expect(result[0]).to.deep.equal(frames[0]);
      expect(result[1]).to.deep.equal(frames[1]);
      expect(result[2]).to.deep.equal(frames[2]);
      expect(result[3]).to.deep.equal(frames[2]);
      expect(result[4]).to.deep.equal(frames[2]);
      expect(result[5]).to.deep.equal(frames[2]);
    });

    it('preserves feature values correctly after downsampling', () => {
      const frames = makeFrames(6, 4);
      const result = normalizeFrames(frames, 3);
      expect(result).to.have.length(3);
      expect(result[0]).to.deep.equal(frames[0]);
      expect(result[1]).to.deep.equal(frames[3]); // round(5*1/2) = 3
      expect(result[2]).to.deep.equal(frames[5]);
    });
  });
});
