class Network {
    constructor(neuronCounts) {
        this.levels = [];
        for (let i = 0; i < neuronCounts.length - 1; i++) {
            this.levels.push(new Level(neuronCounts[i], neuronCounts[i + 1]));
        }
    }

    static feedForward(givenInputs, network) {
        let outputs = givenInputs;
    
        for (let i = 0; i < network.levels.length; i++) {
            const isLast = i === network.levels.length - 1;
            outputs = Level.feedForward(outputs, network.levels[i], isLast);
        }
    
        return outputs;
    }

    static mutate(network, rate = 0.08, strength = 0.12) {
        network.levels.forEach((level, li) => {
            const isOutputLayer = li === network.levels.length - 1;
            const s = isOutputLayer ? strength * 0.4 : strength;
            const r = isOutputLayer ? rate * 0.6 : rate;

            for (let i = 0; i < level.biases.length; i++) {
                if (Math.random() < r) {
                    level.biases[i] += gaussianRandom(0, s);
                    level.biases[i] = Math.max(-2, Math.min(2, level.biases[i]));
                }
            }
            for (let i = 0; i < level.weights.length; i++) {
                for (let j = 0; j < level.weights[i].length; j++) {
                    if (Math.random() < r) {
                        level.weights[i][j] += gaussianRandom(0, s);
                        level.weights[i][j] = Math.max(-2, Math.min(2, level.weights[i][j]));
                    }
                }
            }
        });
    }
}

class Level {
    constructor(inputCount, outputCount) {
        this.inputs = new Array(inputCount);
        this.outputs = new Array(outputCount);
        this.biases = new Array(outputCount);
        this.weights = [];
        for (let i = 0; i < inputCount; i++) {
            this.weights[i] = new Array(outputCount);
        }
        Level.#randomize(this);
    }

    static #randomize(level) {
        const fanIn  = level.inputs.length;
        const fanOut = level.outputs.length;
        const limit  = Math.sqrt(6 / (fanIn + fanOut));

        for (let i = 0; i < level.inputs.length; i++) {
            for (let j = 0; j < level.outputs.length; j++) {
                level.weights[i][j] = (Math.random() * 2 - 1) * limit;
            }
        }
        for (let i = 0; i < level.biases.length; i++) {
            level.biases[i] = 0; 
        }
    }

    static feedForward(inputs, level, isOutputLayer = false) {
        for (let i = 0; i < level.inputs.length; i++) {
            level.inputs[i] = inputs[i] ?? 0;
        }
    
        for (let i = 0; i < level.outputs.length; i++) {
            let sum = 0;
    
            for (let j = 0; j < level.inputs.length; j++) {
                sum += level.inputs[j] * level.weights[j][i];
            }
    
            let activated = sum + level.biases[i];
    
            if (isOutputLayer) {
                level.outputs[i] = sigmoid(activated);
            } else {
                level.outputs[i] = leakyRelu(activated);
            }
        }
    
        return [...level.outputs];
    }
}