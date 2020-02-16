use json;
use neural_network::nn;
use neural_network::nn_trainer;
use std::fs;

fn main() {
    let data_load = load_data();
    let data: Vec<(Vec<f64>, Vec<f64>)>;
    match data_load {
        Ok(v) => data = v,
        Err(msg) => {
            println!("an error occured: {}", msg);
            return;
        }
    }

    let mut nn = nn::NeuralNetwork::new_rand(&vec![784usize, 10usize], 0.1);
    train(&mut nn, &data);

    display_results(&mut nn, &data);
}

fn display_results(nn: &mut nn::NeuralNetwork, data: &Vec<(Vec<f64>, Vec<f64>)>) {
    for i in 1000..1010 {
        let (x, y) = &data[i];
        display_digit(x);
        let pred = nn.forward(x);
        println!("label: {}", get_max_index(y));
        println!("prediction: {:?}", get_max_index(&pred));
        println!("prediction vector: {:?}", pred);
    }
}

fn train(nn: &mut nn::NeuralNetwork, data: &Vec<(Vec<f64>, Vec<f64>)>) {
    println!("training..");
    let mut nn_t = nn_trainer::NeuralNetworkTrainer::new(&nn);

    // batches
    for i in 1..2 {
        // epochs
        for _ in 0..3 {
            let e = nn_t.update_mini_batch(
                nn,
                &data[((i - 1) * 1000)..(i * 1000 - 1)].to_vec(),
                0.5,
                0.1,
            );
            println!("err {}", e);
        }
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

fn load_data() -> Result<Vec<(Vec<f64>, Vec<f64>)>, String> {
    println!("loading data..");
    let data = fs::read_to_string("data.json").map_err(|e| e.to_string())?;
    let res = json::parse(&data).map_err(|e| e.to_string())?;

    let mut data = Vec::new();

    for i in 0..res["x"].len() {
        let mut pic = Vec::new();
        for j in 0..res["x"][i].len() {
            for k in 0..res["x"][i][j].len() {
                let (_, v, _) = res["x"][i][j][k].as_number().ok_or("")?.as_parts();
                pic.push(v as f64);
            }
        }
        let (_, v, _) = res["y"][i].as_number().ok_or("")?.as_parts();
        let mut y = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        y[v as usize] = 1.0;
        data.push((pic, y));
    }
    Ok(data)
}

fn get_max_index(v: &Vec<f64>) -> u32 {
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
