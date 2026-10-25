// Estimates VRM mouth shapes (visemes) from the bot's audio in real time.
//
// Volume decides how open the mouth is; the energy split between three
// frequency bands (roughly where vowel formants F1/F2 sit) picks the shape:
//   250-800 Hz   -> rounded vowels (o, u)
//   800-1400 Hz  -> open vowels (a)
//   1400-3000 Hz -> spread vowels (e, i)

export type Visemes = { aa: number; ih: number; ou: number; ee: number; oh: number };

const ZERO: Visemes = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };

export class LipSync {
  private ctx: AudioContext;
  private source: MediaStreamAudioSourceNode;
  private analyser: AnalyserNode;
  private freq: Float32Array<ArrayBuffer>;
  private wave: Float32Array<ArrayBuffer>;
  private current: Visemes = { ...ZERO };
  private resumeOnGesture = () => this.ctx.resume();

  constructor(track: MediaStreamTrack) {
    this.ctx = new AudioContext();
    this.source = this.ctx.createMediaStreamSource(new MediaStream([track]));
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyser.smoothingTimeConstant = 0.4;
    // Analysis only: playback is handled by PipecatClientAudio
    this.source.connect(this.analyser);
    this.freq = new Float32Array(this.analyser.frequencyBinCount);
    this.wave = new Float32Array(this.analyser.fftSize);
    // Browsers may start the context suspended until the next user gesture
    if (this.ctx.state === "suspended") {
      this.ctx.resume();
      window.addEventListener("pointerdown", this.resumeOnGesture, { once: true });
    }
  }

  update(deltaSecs: number): Visemes {
    this.analyser.getFloatTimeDomainData(this.wave);
    let sumSquares = 0;
    for (const sample of this.wave) sumSquares += sample * sample;
    const rms = Math.sqrt(sumSquares / this.wave.length);
    const volume = clamp((rms - 0.008) * 10, 0, 1);

    const target = { ...ZERO };
    if (volume > 0) {
      this.analyser.getFloatFrequencyData(this.freq);
      const round = this.bandEnergy(250, 800);
      const open = this.bandEnergy(800, 1400);
      const spread = this.bandEnergy(1400, 3000);
      const total = round + open + spread + 1e-9;
      const [r, o, s] = [round / total, open / total, spread / total];

      // Always open the jaw a bit with sound, then shape it by vowel
      target.aa = volume * (0.35 + 0.8 * o);
      target.ee = volume * 0.7 * s;
      target.ih = volume * 0.4 * s;
      target.oh = volume * 0.8 * r;
      target.ou = volume * 0.4 * r;

      const sum = target.aa + target.ee + target.ih + target.oh + target.ou;
      if (sum > 1) for (const key of keys(target)) target[key] /= sum;
    }

    // Open quickly, close a bit slower, so it doesn't flicker between syllables
    for (const key of keys(target)) {
      const rate = target[key] > this.current[key] ? 30 : 15;
      this.current[key] += (target[key] - this.current[key]) * (1 - Math.exp(-rate * deltaSecs));
    }
    return this.current;
  }

  dispose() {
    window.removeEventListener("pointerdown", this.resumeOnGesture);
    this.source.disconnect();
    this.ctx.close();
  }

  private bandEnergy(fromHz: number, toHz: number): number {
    const hzPerBin = this.ctx.sampleRate / this.analyser.fftSize;
    let energy = 0;
    for (let bin = Math.floor(fromHz / hzPerBin); bin < Math.ceil(toHz / hzPerBin); bin++) {
      energy += 10 ** (this.freq[bin] / 10); // dB -> linear power
    }
    return energy;
  }
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const keys = (v: Visemes) => Object.keys(v) as (keyof Visemes)[];
