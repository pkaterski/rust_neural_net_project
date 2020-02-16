use crate::nn;

struct NeuronHelper {
    delta: f64,   // delta(l) = ∂C/∂a(l)
    a: f64,       // a(l) = s(z(l))
    z: f64,       // z(l) = wz(l-1) + b(l)
    db: f64,      //∂C/∂b
    dw: Vec<f64>, // ∂C/∂w
}

impl NeuronHelper {
    fn init(input_count: usize) -> Self {
        let mut dw = Vec::new();
        for _ in 0..input_count {
            dw.push(0.0);
        }
        NeuronHelper {
            delta: 0.0,
            a: 0.0,
            z: 0.0,
            db: 0.0,
            dw,
        }
    }
}

struct LayerHelper {
    neurons: Vec<NeuronHelper>,
}

pub struct NeuralNetworkTrainer {
    // the layers containing the gradient of the previous update to the neural network
    // stored for applying momentum to the learning
    prev_layers: Vec<LayerHelper>,
}

impl NeuralNetworkTrainer {
    pub fn new(nn: &nn::NeuralNetwork) -> Self {
        NeuralNetworkTrainer {
            prev_layers: NeuralNetworkTrainer::init_layers(nn),
        }
    }

    fn init_layers(nn: &nn::NeuralNetwork) -> Vec<LayerHelper> {
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

    fn cost(output: &Vec<f64>, target_output: &Vec<f64>) -> f64 {
        if output.len() != target_output.len() {
            panic!("Cannot compute error. NN output and y dims mismatch");
        }
        let mut total = 0.0;
        for i in 0..output.len() {
            total += (output[i] - target_output[i]) * (output[i] - target_output[i]);
        }
        total / 2.0
    }

    fn cost_prime(output: f64, y: f64) -> f64 {
        output - y
    }

    fn backprop(
        nn: &mut nn::NeuralNetwork,
        input: &Vec<f64>,
        target_output: &Vec<f64>,
    ) -> Vec<LayerHelper> {
        // handle error
        if nn.layers.len() < 2 {
            panic!("invalid neural network");
        }

        if nn.layers[0].neurons.len() != input.len() {
            panic!("nerual network layer_0 and input mismatch");
        }

        if nn.layers[nn.layers.len() - 1].neurons.len() != target_output.len() {
            panic!("nerual network layer_Last and output mismatch");
        }

        // set up the helper layers
        let mut layer_helpers = NeuralNetworkTrainer::init_layers(nn);
        // Set the corresponding activation a1 for the input layer.
        for i in 0..nn.layers[0].neurons.len() {
            nn.layers[0].neurons[i].output = input[i]; //TODO: remove this
            layer_helpers[0].neurons[i].a = input[i];
        }

        // Feedforward: For each l=2,3,…,L compute zl=wla(l−1)+bl and al=σ(zl).
        for i in 1..nn.layers.len() {
            for j in 0..nn.layers[i].neurons.len() {
                let mut total = 0.0;
                for k in 0..nn.layers[i].neurons[j].weights.len() {
                    // TODO: use layer helpers
                    total +=
                        nn.layers[i - 1].neurons[k].output * nn.layers[i].neurons[j].weights[k];
                }
                total += nn.layers[i].neurons[j].bias;
                layer_helpers[i].neurons[j].z = total; // z of helper (before activation)
                total = nn::activate(total, nn.layers[i].activation);
                layer_helpers[i].neurons[j].a = total; // A = σ(z)
                nn.layers[i].neurons[j].output = total;
            }
        }

        // Output error δL: Compute the vector δL=∇aC⊙σ′(zL).
        let last = layer_helpers.len() - 1;

        for i in 0..layer_helpers[last].neurons.len() {
            layer_helpers[last].neurons[i].delta = NeuralNetworkTrainer::cost_prime(
                layer_helpers[last].neurons[i].a,
                target_output[i],
            ) * nn::activate_prime(
                layer_helpers[last].neurons[i].z,
                nn.layers[last].activation,
            );

            // Output: The gradient of the cost function is given by ∂C∂wl and ∂C∂blj=δlj.
            layer_helpers[last].neurons[i].db = layer_helpers[last].neurons[i].delta;
            for j in 0..layer_helpers[last - 1].neurons.len() {
                layer_helpers[last].neurons[i].dw[j] =
                    layer_helpers[last - 1].neurons[j].a * layer_helpers[last].neurons[i].delta;
            }
        }

        // Backpropagate the error: For each l=L−1,L−2,…,2 compute δl=((wl+1)Tδl+1)⊙σ′(zl).
        // δlj=∑k w(l+1)kj δ(l+1)kσ′(zlj).
        for i in (1..(layer_helpers.len() - 1)).rev() {
            for j in 0..layer_helpers[i].neurons.len() {
                let mut total = 0.0;

                for k in 0..layer_helpers[i + 1].neurons.len() {
                    total += layer_helpers[i + 1].neurons[k].delta
                        * nn.layers[i + 1].neurons[k].weights[j]
                        * nn::activate_prime(
                            layer_helpers[i].neurons[j].z,
                            nn.layers[i].activation,
                        );
                }

                layer_helpers[i].neurons[j].delta = total;

                // Output: The gradient of the cost function is given by ∂C∂wl and ∂C∂blj=δlj.
                layer_helpers[i].neurons[j].db = layer_helpers[i].neurons[j].delta;
                for k in 0..layer_helpers[i - 1].neurons.len() {
                    layer_helpers[i].neurons[j].dw[k] =
                        layer_helpers[i - 1].neurons[k].a * layer_helpers[i].neurons[j].delta;
                }
            }
        }
        layer_helpers
    }

    pub fn update_mini_batch(
        &mut self,
        nn: &mut nn::NeuralNetwork,
        mini_batch: &Vec<(Vec<f64>, Vec<f64>)>,
        eta: f64,
        momentum: f64,
    ) -> f64 {
        let mut nabla_layers: Vec<LayerHelper>;
        for data in mini_batch.iter() {
            let (x, y) = data;
            nabla_layers = NeuralNetworkTrainer::backprop(nn, x, y);

            // update weights and biases
            for i in 0..nn.layers.len() {
                for j in 0..nn.layers[i].neurons.len() {
                    nn.layers[i].neurons[j].bias -=
                        (1.0 - momentum) * nabla_layers[i].neurons[j].db * eta / 1.0
                            + momentum * self.prev_layers[i].neurons[j].db;

                    self.prev_layers[i].neurons[j].db = nabla_layers[i].neurons[j].db * eta / 1.0;

                    for k in 0..nn.layers[i].neurons[j].weights.len() {
                        nn.layers[i].neurons[j].weights[k] -=
                            (1.0 - momentum) * nabla_layers[i].neurons[j].dw[k] * eta / 1.0
                                + momentum * self.prev_layers[i].neurons[j].dw[k];

                        self.prev_layers[i].neurons[j].dw[k] =
                            nabla_layers[i].neurons[j].dw[k] * eta / 1.0;
                    }
                }
            }
        }
        // return the error
        let mut error_value = 0.0;
        for i in 0..mini_batch.len() {
            let (x, y) = &mini_batch[i];
            error_value += NeuralNetworkTrainer::cost(&nn.forward(&x), y);
        }
        error_value / (mini_batch.len() as f64)
    }
}
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_derivative_sigmoid() {
        let x = 4.3;
        let delta = 0.00001;

        let f = crate::nn::activate(x, crate::nn::Activation::Sigmoid);
        let f_d = crate::nn::activate(x + delta, crate::nn::Activation::Sigmoid);
        let f_p = crate::nn::activate_prime(x, crate::nn::Activation::Sigmoid);

        let f_p2 = (f_d - f) / delta;

        assert!((f_p - f_p2).abs() < 0.001);
    }

    #[test]
    fn test_logical_or() {
        let data = vec![
            (vec![0.0, 0.0], vec![1.0, 0.0]),
            (vec![0.0, 1.0], vec![0.0, 1.0]),
            (vec![1.0, 0.0], vec![0.0, 1.0]),
            (vec![1.0, 1.0], vec![0.0, 1.0]),
        ];

        let mut nn = crate::nn::NeuralNetwork::new_rand(&vec![2usize, 2usize], 0.1);
        let mut nn_t = NeuralNetworkTrainer::new(&mut nn);

        for _ in 0..1000 {
            let e = nn_t.update_mini_batch(&mut nn, &data, 0.5, 0.1);
            println!("{}", e);
            if e < 0.001 {
                break;
            }
        }
        assert_eq!(get_max_index(nn.forward(&vec![0.0, 0.0])), 0);
        assert_eq!(get_max_index(nn.forward(&vec![0.0, 1.0])), 1);
        assert_eq!(get_max_index(nn.forward(&vec![1.0, 0.0])), 1);
        assert_eq!(get_max_index(nn.forward(&vec![1.0, 1.0])), 1);
    }

    #[test]
    fn test_logical_and() {
        let data = vec![
            (vec![0.0, 0.0], vec![1.0, 0.0]),
            (vec![0.0, 1.0], vec![1.0, 0.0]),
            (vec![1.0, 0.0], vec![1.0, 0.0]),
            (vec![1.0, 1.0], vec![0.0, 1.0]),
        ];

        let mut nn = crate::nn::NeuralNetwork::new_rand(&vec![2usize, 2usize], 0.1);
        let mut nn_t = NeuralNetworkTrainer::new(&mut nn);

        for _ in 0..1000 {
            let e = nn_t.update_mini_batch(&mut nn, &data, 0.5, 0.1);
            println!("{}", e);
            if e < 0.001 {
                break;
            }
        }
        assert_eq!(get_max_index(nn.forward(&vec![0.0, 0.0])), 0);
        assert_eq!(get_max_index(nn.forward(&vec![0.0, 1.0])), 0);
        assert_eq!(get_max_index(nn.forward(&vec![1.0, 0.0])), 0);
        assert_eq!(get_max_index(nn.forward(&vec![1.0, 1.0])), 1);
    }

    fn get_max_index(v: Vec<f64>) -> u32 {
        if v.len() == 0 {
            panic!("oh no - cannot get max of empty vector");
        }
        let mut index = 0;
        let mut max = v[0];
        for i in 0..v.len() {
            if v[i] > max {
                max = v[i];
                index = i;
            }
        }
        index as u32
    }

    #[test]
    #[should_panic(expected = "Cannot compute error. NN output and y dims mismatch")]
    fn cost_dimentions_mismatch() {
        let y = vec![1.0, 0.0];
        let y_t = vec![1.0];

        NeuralNetworkTrainer::cost(&y, &y_t);
    }

    #[test]
    #[should_panic(expected = "nerual network layer_0 and input mismatch")]
    fn trainer_input_mismatch() {
        let dim = vec![10usize, 2usize];
        let data = vec![
            (vec![0.0, 0.0], vec![1.0, 0.0]),
            (vec![0.0, 1.0], vec![1.0, 0.0]),
            (vec![1.0, 0.0], vec![1.0, 0.0]),
            (vec![1.0, 1.0], vec![0.0, 1.0]),
        ];

        let mut nn = crate::nn::NeuralNetwork::new_rand(&dim, 0.1);
        let mut nn_t = NeuralNetworkTrainer::new(&mut nn);

        nn_t.update_mini_batch(&mut nn, &data, 0.5, 0.1);
    }

    #[test]
    #[should_panic(expected = "nerual network layer_Last and output mismatch")]
    fn trainer_output_mismatch() {
        let dim = vec![2usize, 20usize];
        let data = vec![
            (vec![0.0, 0.0], vec![1.0, 0.0]),
            (vec![0.0, 1.0], vec![1.0, 0.0]),
            (vec![1.0, 0.0], vec![1.0, 0.0]),
            (vec![1.0, 1.0], vec![0.0, 1.0]),
        ];

        let mut nn = crate::nn::NeuralNetwork::new_rand(&dim, 0.1);
        let mut nn_t = NeuralNetworkTrainer::new(&mut nn);

        nn_t.update_mini_batch(&mut nn, &data, 0.5, 0.1);
    }
}
