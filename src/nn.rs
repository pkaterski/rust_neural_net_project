use rand::Rng;

pub struct Neuron {
    pub bias: f64,
    pub output: f64,
    pub weights: Vec<f64>,
}

impl Neuron {
    // fn new(weights: Vec<f64>, bias: f64) -> Self {
    //     Neuron { weights, bias, output: 0.0 }
    // }

    fn new_rand(input_count: usize, mut rng: rand::rngs::ThreadRng, scale: f64) -> Self {
        let mut weights = Vec::new();
        for _ in 0..input_count {
            weights.push(rng.gen::<f64>() * scale);
        }
        let bias = rng.gen();
        let output = 0.0;
        Neuron {
            weights,
            bias,
            output,
        }
    }
}

#[derive(Copy, Clone)]
pub enum Activation {
    Sigmoid,
    ReLU,
    Id,
    Custom {
        activate: fn(f64) -> f64,
        activate_prime: fn(f64) -> f64,
    },
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub activation: Activation,
}

impl Layer {
    // fn new(neurons: Vec<Neuron>, activation: Activation) -> Self {
    //     Layer { neurons, activation }
    // }

    fn new_rand(
        size: usize,
        prev_layer_size: usize,
        rng: rand::rngs::ThreadRng,
        scale: f64,
        activation: Activation,
    ) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..size {
            neurons.push(Neuron::new_rand(prev_layer_size, rng, scale));
        }
        Layer {
            neurons,
            activation,
        }
    }

    fn size(&self) -> usize {
        self.neurons.len()
    }

    fn get_values(&self) -> Vec<f64> {
        self.neurons.iter().map(|n| n.output).collect()
    }
}

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new_rand(sizes: &Vec<usize>, scale: f64) -> Self {
        let mut layers = Vec::new();
        let sigmoid = Activation::Sigmoid;
        let rng = rand::thread_rng();

        layers.push(Layer::new_rand(sizes[0], 0, rng, 0.0, Activation::Id));
        for i in 0..layers[0].neurons.len() {
            layers[0].neurons[i].bias = 0.0;
        }

        for i in 1..sizes.len() {
            layers.push(Layer::new_rand(sizes[i], sizes[i - 1], rng, scale, sigmoid));
        }

        NeuralNetwork { layers }
    }

    pub fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        if self.layers.len() == 0 {
            panic!("no initial layer");
        }

        if input.len() != self.layers[0].size() {
            panic!("input and layer_0 dimimentions mismatch");
        }

        for i in 0..input.len() {
            self.layers[0].neurons[i].output = input[i];
        }

        for i in 1..self.layers.len() {
            for j in 0..self.layers[i].neurons.len() {
                let mut total = 0.0;

                for k in 0..self.layers[i - 1].neurons.len() {
                    total +=
                        self.layers[i - 1].neurons[k].output * self.layers[i].neurons[j].weights[k];
                }

                total += self.layers[i].neurons[j].bias;
                total = activate(total, self.layers[i].activation);
                self.layers[i].neurons[j].output = total;
            }
        }
        self.layers[self.layers.len() - 1].get_values()
    }
}

pub fn activate(x: f64, activation: Activation) -> f64 {
    match activation {
        Activation::Sigmoid => 1.0 / (1.0 + (1.0_f64).exp().powf(-x)),
        Activation::ReLU => {
            if x >= 0.0 {
                x
            } else {
                0.0
            }
        }
        Activation::Id => x,
        Activation::Custom {
            activate: s,
            activate_prime: _,
        } => s(x),
    }
}

pub fn activate_prime(x: f64, activation: Activation) -> f64 {
    match activation {
        Activation::Sigmoid => {
            let s = activate(x, Activation::Sigmoid);
            s * (1.0 - s)
        }
        Activation::ReLU => {
            if x >= 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Activation::Id => x,
        Activation::Custom {
            activate: _,
            activate_prime: s,
        } => s(x),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_initial_layer() {
        let nn = NeuralNetwork::new_rand(&vec![888, 20, 10], 1.0);
        assert_eq!(nn.layers[0].neurons.len(), 888);
        for i in 0..nn.layers[0].neurons.len() {
            match nn.layers[0].activation {
                Activation::Id => (),
                _ => panic!("non id activation in initial layer"),
            }
            assert_eq!(nn.layers[0].neurons[i].bias, 0.0);
        }
    }

    #[test]
    fn test_layer_size() {
        let nn = NeuralNetwork::new_rand(&vec![888, 20, 55, 3, 7, 10], 1.0);
        assert_eq!(nn.layers.len(), 6);
    }

    #[test]
    fn test_neurons_length() {
        let dim = vec![888, 20, 55, 3, 7, 10];
        let nn = NeuralNetwork::new_rand(&dim, 1.0);
        for i in 0..dim.len() {
            assert_eq!(nn.layers[i].neurons.len(), dim[i]);
        }
    }

    #[test]
    fn test_weights_ranges() {
        let dim = vec![888, 20, 55, 3, 7, 10];
        let nn = NeuralNetwork::new_rand(&dim, 1.0);
        for i in 0..dim.len() {
            for j in 0..nn.layers[i].neurons.len() {
                for k in 0..nn.layers[i].neurons[j].weights.len() {
                    assert!(nn.layers[i].neurons[j].weights[k] < 1.0);
                    assert!(nn.layers[i].neurons[j].weights[k] >= 0.0);
                }
            }
        }
    }

    #[test]
    fn test_forward() {
        let dim = vec![4, 20, 55, 3, 7, 10];
        let mut nn = NeuralNetwork::new_rand(&dim, 1.0);
        let res = nn.forward(&vec![0.0, 0.5, 0.8, 0.9]);
        assert_eq!(res.len(), 10);
    }
}
