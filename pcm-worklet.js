class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ws = null;
    this.isSocketOpen = false;
    this.textDecoder = new TextDecoder();
    this.port.onmessage = (event) => {
      this.handleControlMessage(event.data);
    };
  }

  handleControlMessage(message) {
    if (!message || typeof message !== "object") return;

    if (message.type === "ws_connect") {
      this.connectSocket(message.url);
      return;
    }
    if (message.type === "ws_close") {
      this.closeSocket();
      return;
    }
    if (message.type === "ws_send_text" && typeof message.payload === "string") {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(message.payload);
      }
    }
  }

  connectSocket(url) {
    if (!url || typeof url !== "string") return;

    if (typeof WebSocket === "undefined") {
      this.port.postMessage({
        type: "ws_error",
        message: "WebSocket is not available in AudioWorkletGlobalScope",
      });
      return;
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }

    this.closeSocket();

    try {
      this.ws = new WebSocket(url);
      this.ws.binaryType = "arraybuffer";
    } catch (error) {
      this.port.postMessage({
        type: "ws_error",
        message: error && error.message ? error.message : String(error),
      });
      return;
    }

    this.ws.onopen = () => {
      this.isSocketOpen = true;
      this.port.postMessage({ type: "ws_status", status: "open" });
    };

    this.ws.onclose = () => {
      this.isSocketOpen = false;
      this.ws = null;
      this.port.postMessage({ type: "ws_status", status: "closed" });
    };

    this.ws.onerror = () => {
      this.port.postMessage({ type: "ws_error", message: "runtime error" });
    };

    this.ws.onmessage = (event) => {
      let payload = event.data;
      if (payload instanceof ArrayBuffer) {
        payload = this.textDecoder.decode(payload);
      }
      if (typeof payload !== "string") {
        payload = String(payload);
      }
      this.port.postMessage({ type: "ws_message", payload });
    };
  }

  closeSocket() {
    if (this.ws) {
      const socket = this.ws;
      this.ws = null;
      this.isSocketOpen = false;
      socket.onopen = null;
      socket.onclose = null;
      socket.onmessage = null;
      socket.onerror = null;
      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
        socket.close();
      }
    }
  }

  process(inputs) {
    const input = inputs[0];

    if (input && input[0] && this.ws && this.ws.readyState === WebSocket.OPEN) {
      const channelData = input[0];
      const pcm16 = new Int16Array(channelData.length);

      for (let i = 0; i < channelData.length; i++) {
        const sample = Math.max(-1, Math.min(1, channelData[i]));
        pcm16[i] = sample < 0 ? sample * 32768 : sample * 32767;
      }

      this.ws.send(pcm16.buffer);
    }

    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
