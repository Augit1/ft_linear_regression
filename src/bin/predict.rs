use std::{error::Error, fs};
use std::io::{self, Write};

use ft_linear_regression::{Model, SCALE_X};

fn main() -> Result<(), Box<dyn Error>> {
    let m: Model = serde_json::from_str(&fs::read_to_string("theta.json")?)?;

    print!("Mileage ? ");
    io::stdout().flush()?;
    let mut s = String::new();
    io::stdin().read_line(&mut s)?;
    let mileage: f64 = s.trim().parse()?;

    let x = mileage / SCALE_X;
    let price = (m.theta0 + m.theta1 * x).max(0.0);

    println!("{:.2}€", price);
    Ok(())
}
