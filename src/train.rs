use std::{error::Error, fs, fs::File};
use csv::Reader;
use serde::{Deserialize, Serialize};
use plotters::prelude::*;
use plotters::prelude::IntoLogRange;

const SCALE_X: f64 = 100_000.0; // même constante côté predict.rs

#[derive(Serialize, Deserialize)]
struct Model { theta0: f64, theta1: f64 }

// === I/O =====================================================================

fn read_csv_raw(path: &str) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let mut rdr = Reader::from_reader(File::open(path)?);
    let mut x_km = Vec::new();
    let mut y = Vec::new();
    // csv::Reader::records() saute l'en-tête par défaut
    for rec in rdr.records() {
        let r = rec?;
        let mileage_km: f64 = r[0].parse()?;
        let price:      f64 = r[1].parse()?;
        x_km.push(mileage_km); // <-- km bruts, pas de scaling ici
        y.push(price);
    }
    Ok((x_km, y))
}

fn save_model(m: &Model, path: &str) -> Result<(), Box<dyn Error>> {
    fs::write(path, serde_json::to_string_pretty(m)?)?;
    Ok(())
}

// === ML ======================================================================

fn train(x_scaled: &[f64], y: &[f64], lr: f64, iters: usize) -> (f64, f64, Vec<f64>) {
    let m = x_scaled.len();
    assert!(m == y.len() && m > 0, "empty or mismatched data");
    let inv_m = 1.0 / (m as f64);

    let (mut t0, mut t1) = (0.0, 0.0);
    let mut history = Vec::with_capacity(iters);

    for _ in 0..iters {
        // loss BEFORE update (courbe commence à l'itération 0)
        let loss = x_scaled.iter().zip(y)
            .map(|(xi, yi)| { let e = (t0 + t1 * xi) - yi; e*e })
            .sum::<f64>() * inv_m;
        history.push(loss);
        if !t0.is_finite() || !t1.is_finite() || !loss.is_finite() { break; }

        // gradients
        let (sum0, sum1) = x_scaled.iter().zip(y).fold((0.0, 0.0), |(a0, a1), (xi, yi)| {
            let err = (t0 + t1 * xi) - yi;
            (a0 + err, a1 + err * xi)
        });

        // simultaneous update
        t0 -= lr * (sum0 * inv_m);
        t1 -= lr * (sum1 * inv_m);
    }
    (t0, t1, history)
}

// === Plots ===================================================================


fn plot_loss(history: &[f64], out_svg: &str) -> Result<(), Box<dyn std::error::Error>> {
    if history.is_empty() { return Ok(()); }

    let rmse_raw: Vec<f64> = history.iter().map(|&m| m.sqrt()).collect();

    let eps = 1e-9;
    let rmse: Vec<f64> = rmse_raw.iter().map(|&v| v.max(eps)).collect();

    let mut ymin = rmse.iter().copied().fold(f64::INFINITY, f64::min);
    let mut ymax = rmse.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (ymax - ymin).abs() < f64::EPSILON {
        ymin = (ymin / 2.0).max(eps);
        ymax = ymax * 2.0;
    }

    let root = SVGBackend::new(out_svg, (900, 500)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(12)
        .caption("Evolution of Prediction Error (RMSE)", ("sans-serif", 22))
        .x_label_area_size(40)
        .y_label_area_size(58)
        .build_cartesian_2d(0..(rmse.len() - 1), (ymin..ymax).log_scale())?;

    chart.configure_mesh()
        .x_desc("Training Iteration")
        .y_desc("Average Prediction Error (RMSE, in €)")
        .y_labels(12)
        .y_label_formatter(&|v| format!("{:.0}", v))
        .axis_style(&BLACK.mix(0.6))
        .light_line_style(&BLACK.mix(0.06)) 
        .label_style(("sans-serif", 13))
        .draw()?;

    chart.draw_series(LineSeries::new(
        rmse.iter().enumerate().map(|(i, &v)| (i, v)),
        &BLUE,
    ))?;

    Ok(())
}

fn plot_fit(
    x_km: &[f64],
    y: &[f64],
    theta0: f64,
    theta1: f64,
    out_svg: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if x_km.is_empty() || x_km.len() != y.len() {
        return Ok(()); // rien à tracer ou données incohérentes
    }

    // bornes brutes
    let xmin = x_km.iter().copied().fold(f64::INFINITY, f64::min);
    let xmax = x_km.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ymin = y.iter().copied().fold(f64::INFINITY, f64::min);
    let ymax = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // range + padding (5%)
    let xr = (xmax - xmin).abs();
    let yr = (ymax - ymin).abs();
    let xr = if xr == 0.0 { 1.0 } else { xr };
    let yr = if yr == 0.0 { 1.0 } else { yr };

    let xlo = xmin - xr * 0.05;
    let xhi = xmax + xr * 0.05;
    let ylo = ymin - yr * 0.05;
    let yhi = ymax + yr * 0.05;

    // canvas
    let root = SVGBackend::new(out_svg, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(12)
        .caption("Price vs Mileage (fit)", ("sans-serif", 22))
        .x_label_area_size(36)
        .y_label_area_size(56)
        .build_cartesian_2d(xlo..xhi, ylo..yhi)?;

    chart.configure_mesh()
        .x_desc("Mileage (km)")
        .y_desc("Price")
        .axis_style(&BLACK.mix(0.6))
        .light_line_style(&BLACK.mix(0.06))
        .label_style(("sans-serif", 13))
        .draw()?;

    // --- Ligne de régression (dessinée d'abord) ---
    // y = θ0 + θ1 * (x_km / SCALE_X)
    let y1 = theta0 + theta1 * (xlo / SCALE_X);
    let y2 = theta0 + theta1 * (xhi / SCALE_X);
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(xlo, y1), (xhi, y2)],
        ShapeStyle {
            color: RED.mix(0.9),
            filled: false,
            stroke_width: 2,
        },
    )))?
    .label("Linear fit")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));

    // --- Nuage de points (par-dessus) ---
    chart.draw_series(
        x_km.iter().zip(y).map(|(xr, yr)| Circle::new((*xr, *yr), 3, BLUE.filled()))
    )?
    .label("Data")
    .legend(|(x, y)| Circle::new((x, y), 3, BLUE.filled()));

    chart.configure_series_labels()
        .border_style(&BLACK.mix(0.4))
        .background_style(&WHITE.mix(0.8))
        .label_font(("sans-serif", 12))
        .draw()?;

    Ok(())
}

// === Main ====================================================================

fn main() -> Result<(), Box<dyn Error>> {
    // 1) km BRUTS
    let (x_km, y) = read_csv_raw("data.csv")?;

    // 2) feature scalée pour train
    let x: Vec<f64> = x_km.iter().map(|&km| km / SCALE_X).collect();

    // 3) train + historique
    let (theta0, theta1, history) = train(&x, &y, 1e-2, 10_000);
    println!("θ0 = {theta0}, θ1 = {theta1}");

    // 4) save + plots (PDF vectoriel)
    save_model(&Model { theta0, theta1 }, "theta.json")?;
    plot_fit(&x_km, &y, theta0, theta1, "fit.svg")?;
    plot_loss(&history, "loss.svg")?;
    Ok(())
}
