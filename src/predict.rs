use std::{error::Error, fs};
use serde::Deserialize;

const SCALE_X: f64 = 100_000.0; 

#[derive(Deserialize)]
struct Model { theta0: f64, theta1: f64 }

fn main() -> Result<(), Box<dyn Error>> {
    let m: Model = serde_json::from_str(&fs::read_to_string("theta.json")?)?;

    use std::io::{self, Write};
    print!("Mileage ? ");
    io::stdout().flush()?;
    let mut s = String::new();
    io::stdin().read_line(&mut s)?;
    let mileage: f64 = s.trim().parse()?;

    let x = mileage / SCALE_X;
    let price = (m.theta0 + m.theta1 * x).max(0.0);

    println!("{price}");
    Ok(())
}
