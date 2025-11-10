use std::{error::Error, fs, fs::File};
use csv::Reader;
use serde::{Deserialize, Serialize};



fn read_csv<P: AsRef<Path>>(filename: P) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let file = File::open(filename)?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();

    for result in rdr.records() {
        let record = result?;

        let mileage: f64 = record[0].parse()?;
        let price: f64 = record[1].parse()?;
        x_vals.push(mileage);
        y_vals.push(price);
    }
    Ok((x_vals, y_vals))
}

fn normalize(data: &[f64]) -> (Vec<f64>, f64, f64) {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let norm = data.iter().map(|v| (v - min) / (max - min)).collect();
    (norm, min, max)
}

fn estimate_price(mileage: f64, theta0: f64, theta1: f64) -> f64 {
    theta0 + theta1 * mileage
}

fn mse(x: &[f64], y: &[f64], theta0: f64, theta1: f64) -> f64 {
    let m = x.len() as f64;
    let sum_sq_error: f64 = x.iter().zip(y).map(|(xi, yi)| {
        let err = estimate_price(*xi, theta0, theta1) - yi;
        err * err
    }).sum();
    sum_sq_error / m
}

fn train(x: &[f64], y: &[f64], learning_rate: f64, iterations: usize) -> (f64, f64) {
    let m = x.len() as f64;
    let mut theta0 = 0.0;
    let mut theta1 = 0.0;

    for i in 0..iterations {
        let (sum0, sum1) = x.iter().zip(y).fold((0.0, 0.0), |(acc0, acc1), (xi, yi)| {
            let err = estimate_price(*xi, theta0, theta1) - yi;
            (acc0 + err, acc1 + err * xi)
        });
   
        let tmp0 = learning_rate * (sum0 / m);
        let tmp1 = learning_rate * (sum1 / m);

        theta0 -= tmp0;
        theta1 -= tmp1;

        let loss = mse(x, y, theta0, theta1);
        if !theta0.is_finite() || !theta1.is_finite() || !loss.is_finite() {
            eprintln!("Divergence détectée: réduis le learning_rate (ex. /10).");
            break;
        }
        if i % 1000 == 0 {
            println!("iter {:5} θ0={:.6}, θ1={:.6}, mse={:.6}", i, theta0, theta1, loss);
        }
    }
    (theta0, theta1)
}

use serde::{Serialize, Deserialize};
use std::fs;

#[derive(Serialize, Deserialize)]
struct Model {
    theta0: f64,
    theta1: f64,
}

fn save_model(model: &Model, path: &str) -> Result<(), Box<dyn Error>> {
    let data = serde_json::to_string_pretty(model)?;
    fs::write(path, data)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let (x_vals, y_vals) = read_csv("data.csv")?;

    let learning_rate = 1e-5;
    let iterations = 10_000;

    let (theta0, theta1) = train(&x_vals, &y_vals, learning_rate, iterations);

    println!("Training finished: θ0 = {}, θ1 = {}", theta0, theta1);

    let model = Model { theta0, theta1 };
    save_model(&model, "theta.json")?;
    Ok(())
}
