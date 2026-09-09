class PCMProcessor extends AudioWorkletProcessor {
    process(inputs, outputs) {
        const input = inputs[0];

        if (input && input.length > 0) {
            const channel = input[0];

            if (channel) {
                this.port.postMessage(channel);
            }
        }

        return true;
    }
}

registerProcessor("pcm-processor", PCMProcessor);