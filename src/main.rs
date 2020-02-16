use neural_network::nn;
use neural_network::nn_trainer;
use std::fs;
use json;


fn main() {
    let data = load_data();
    let (x, y) = &data[8];
    display_digit(&x);
    println!("{:?}", y);

    let mut nn = nn::NeuralNetwork::new_rand(&vec![784usize, 10usize], 0.1);
    let mut nn_t = nn_trainer::NeuralNetworkTrainer::new(&nn);

    for i in 1..2 {
        // epochs
        for _ in 0..3 {
            let e = nn_t.update_mini_batch(
                &mut nn,
                &data[((i-1)*1000)..(i*1000 - 1)].to_vec(),
                0.5,
                0.1);
            println!("err {}", e);
        }
    }

    for i in 1000..1010 {
        let (x, y) = &data[i];
        display_digit(x);
        println!("y: {:?}", y);
        println!("predic: {:?}", nn.forward(x));
    }
}

fn display_digit(arr: &Vec<f64>) {
    for i in 0..784 {
        print!("{}", arr[i] as u32);
        if (i + 1) % 28 == 0 {
            println!();
        } 
    }
}

fn load_data() -> Vec<(Vec<f64>, Vec<f64>)> {
    let data = fs::read_to_string("data.json").expect("do you have the file?");
    let res = json::parse(&data).unwrap();

    let mut data = Vec::new();

    for i in 0..res["x"].len() {
        let mut pic = Vec::new();
        for j in 0..res["x"][i].len() {
            for k in 0..res["x"][i][j].len() {
                let (_,v,_) = res["x"][i][j][k].as_number().unwrap().as_parts(); 
                pic.push(v as f64);
            }
        }
        let (_,v,_) = res["y"][i].as_number().unwrap().as_parts();
        let mut y = vec![0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
        y[v as usize] = 1.0;
        data.push((pic, y));
    }
    data
}
