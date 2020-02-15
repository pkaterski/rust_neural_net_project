use crate::nn;

struct NeuronHelper {
    delta: f64, // delta(l) = ∂C/∂a(l)
    a: f64, // a(l) = s(z(l))
    z: f64, // z(l) = wz(l-1) + b(l)
    db: f64, //∂C/∂b
    dw: Vec<f64> // ∂C/∂w
}

impl NeuronHelper {
    fn init(input_count: usize) -> Self {
        let mut dw = Vec::new();
        for i in 0..input_count {
            dw.push(0.0);
        }
        NeuronHelper { delta:0.0, a: 0.0, z: 0.0, db: 0.0, dw }
    }
}

struct LayerHelper {
    neurons: Vec<NeuronHelper>
}

struct NeuralNetworkTrainer {
    // the layers containing the gradient of the previous update to the neural network
    // stored for applying momentum to the learning
    prev_layers: Vec<LayerHelper> 
}

impl NeuralNetworkTrainer {
    fn new(&self, nn: nn::NeuralNetwork) -> Self {
        NeuralNetworkTrainer { prev_layers: NeuralNetworkTrainer::init_layers(nn) }
    }

    fn init_layers(nn: nn::NeuralNetwork) -> Vec<LayerHelper>{
        let mut layer_helpers = Vec::new();

        for i in 0..nn.layers.len() {
            let mut neurons = Vec::new();
            for j in 0..nn.layers[i].neurons.len() {
                neurons.push(NeuronHelper::init(nn.layers[i].neurons[j].weights.len()))
            }
            layer_helpers.push(LayerHelper { neurons })
        }
        layer_helpers
    }

    fn cost(output: Vec<f64>, target_output: Vec<f64>) -> f64 {
        if (output.len() != target_output.len()) {
            panic!("Cannot compute error. NN output and y dims mismatch");
        }
        let mut total = 0.0;
        for i in 0..output.len() {
            total += (output[i] - target_output[i]) * (output[i] - target_output[i]);
        }
        total / 2.0
    }
}
