use neural_network::nn;

fn main() {
    let dim = vec![4,55,3,7,10];
    let mut nn = nn::NeuralNetwork::new_rand(&dim, 1.0);
    let res = nn.forward(&vec![0.0,0.5,0.8,0.9]);

    println!("{:?}", res);
}
