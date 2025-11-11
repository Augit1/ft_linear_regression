use std::{error::Error, fs, fs::File};
use csv::Reader;
use serde::{Deserialize, Serialize};
use plotters::prelude::*;
use plotters::prelude::IntoLogRange;

const SCALE_X: f64 = 100_000.0;
const ITERATIONS: usize = 5_000; 

type DynResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Serialize, Deserialize)]
struct Model { theta0: f64, theta1: f64 }

// === I/O =====================================================================

fn read_csv_raw(path: &str) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let mut rdr = Reader::from_reader(File::open(path)?);
    let mut x_km = Vec::new();
    let mut y = Vec::new();
    
    for rec in rdr.records() {
        let r = rec?;
        let mileage_km: f64 = r[0].parse()?;
        let price:      f64 = r[1].parse()?;
        x_km.push(mileage_km);
        y.push(price);
    }
    Ok((x_km, y))
}

fn save_model(m: &Model, path: &str) -> Result<(), Box<dyn Error>> {
    fs::write(path, serde_json::to_string_pretty(m)?)?;
    Ok(())
}

fn save_frames(x_km: &[f64], y: &[f64], t0: f64, t1: f64, iter: usize) -> DynResult {
    fs::create_dir_all("frames")?;
    if iter % 1000 == 0 {
        let filename = format!("frames/iter_{:05}.svg", iter);
        plot_fit(&x_km, &y, t0, t1, &filename)?;
    }
    Ok(())
}

fn live_preview(x_km: &[f64], y: &[f64], t0: f64, t1: f64, iter: usize) -> DynResult {
     let filename = "live_fit.svg";
     plot_fit(&x_km, &y, t0, t1, &filename)?;
     open::that(filename)?;
     println!("Iter number: {:?}\t θ0: {:?}\t θ1: {:?}", iter, t0, t1);
     Ok(())
}

// === ML ======================================================================

fn train(x_km: &[f64], y: &[f64], lr: f64, iters: usize) -> Result<(f64, f64, Vec<f64>), Box<dyn std::error::Error>> {
    let x_scaled: Vec<f64> = x_km.iter().map(|&km| km / SCALE_X).collect();
    let m = x_scaled.len();

    assert!(m == y.len() && m > 0, "empty or mismatched data");
    let inv_m = 1.0 / (m as f64);

    let (mut t0, mut t1) = (0.0, 0.0);
    let mut evo_t1 = 0.0;
    let mut history = Vec::with_capacity(iters);

    for i in 0..=iters {
        let loss = x_scaled.iter().zip(y)
            .map(|(xi, yi)| { let e = (t0 + t1 * xi) - yi; e*e })
            .sum::<f64>() * inv_m;
        history.push(loss);
        if !t0.is_finite() || !t1.is_finite() || !loss.is_finite() { break; }

        let (sum0, sum1) = x_scaled.iter().zip(y).fold((0.0, 0.0), |(a0, a1), (xi, yi)| {
            let err = (t0 + t1 * xi) - yi;
            (a0 + err, a1 + err * xi)
        });

        t0 -= lr * (sum0 * inv_m);
        t1 -= lr * (sum1 * inv_m);
        save_frames(x_km, y, t0, t1, i)?;
        if ((t1 - evo_t1).abs() / evo_t1.abs().max(1e-12)) > 0.03 {
            live_preview(x_km, y, t0, t1, i)?;
            evo_t1 = t1;
        }
    }
    Ok((t0, t1, history))
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

    let root = SVGBackend::new(out_svg, (1000, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(42)
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
        return Ok(());
    }

    let xmin = x_km.iter().copied().fold(f64::INFINITY, f64::min);
    let xmax = x_km.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ymin = y.iter().copied().fold(f64::INFINITY, f64::min);
    let ymax = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let xr = (xmax - xmin).abs();
    let yr = (ymax - ymin).abs();
    let xr = if xr == 0.0 { 1.0 } else { xr };
    let yr = if yr == 0.0 { 1.0 } else { yr };

    let xlo = xmin - xr * 0.05;
    let xhi = xmax + xr * 0.05;
    let ylo = ymin - yr * 0.05;
    let yhi = ymax + yr * 0.05;

    let root = SVGBackend::new(out_svg, (1000, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(42)
        .caption("Price vs Mileage (fit)", ("sans-serif", 22))
        .x_label_area_size(36)
        .y_label_area_size(76)
        .build_cartesian_2d(xlo..xhi, ylo..yhi)?;

    chart.configure_mesh()
        .x_desc("Mileage (km)")
        .y_desc("Price")
        .axis_style(&BLACK.mix(0.6))
        .light_line_style(&BLACK.mix(0.06))
        .label_style(("sans-serif", 13))
        .draw()?;

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

    // 3) train + historique
    let (theta0, theta1, history) = train(&x_km, &y, 1e-2, ITERATIONS)?;
    println!("θ0 = {theta0}, θ1 = {theta1}");

    // 4) save + plots (PDF vectoriel)
    save_model(&Model { theta0, theta1 }, "theta.json")?;
    plot_fit(&x_km, &y, theta0, theta1, "fit.svg")?;
    plot_loss(&history, "loss.svg")?;
    open::that("loss.svg")?;
    Ok(())
}
