class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
  }

  process(inputs) {
    const input = inputs[0];

    if (input && input[0]) {
      const channelData = input[0];
      const pcm16 = new Int16Array(channelData.length);

      for (let i = 0; i < channelData.length; i++) {
        const sample = Math.max(-1, Math.min(1, channelData[i]));
        pcm16[i] = sample < 0 ? sample * 32768 : sample * 32767;
      }
      this.port.postMessage({ type: "pcm_chunk", payload: pcm16.buffer }, [pcm16.buffer]);
    }

    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
