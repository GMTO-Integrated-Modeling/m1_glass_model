use complot as plt;
use csv;
use serde;
use serde::Deserialize;
use zernike;

#[derive(Debug, Deserialize)]
struct Record {
    #[serde(rename = "Displacement[i] (m)")]
    pub delta_x: f64,
    #[serde(rename = "Displacement[j] (m)")]
    pub delta_y: f64,
    #[serde(rename = "Displacement[k] (m)")]
    pub delta_z: f64,
    #[serde(rename = "X (m)")]
    pub x: f64,
    #[serde(rename = "Y (m)")]
    pub y: f64,
    #[serde(rename = "Z (m)")]
    pub z: f64,
}
fn main() {
    let mut rdr = csv::Reader::from_path("../m1_offaxis_front_disp_test.csv").unwrap();
    let mut data: Vec<Record> = vec![];
    for result in rdr.deserialize() {
        data.push(result.unwrap());
    }
    let n_node = data.len();
    println!("{} records!", n_node);

    let (x, yz): (Vec<f64>, Vec<(f64, f64)>) = data
        .iter()
        .map(|d| (d.x + d.delta_x, (d.y + d.delta_y, d.delta_z)))
        .unzip();
    let (y, z): (Vec<f64>, Vec<f64>) = yz.into_iter().unzip();

    let minmax = |z: &[f64]| {
        let z_max = 1e9 * z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z_min = 1e9 * z.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("Surf. min/max: [{:.0};{:.0}]nm", z_min, z_max);
    };
    minmax(&z);
    let fig = plt::png_canvas("m1_offaxis_front_disp.png");
    //let fig = plt::canvas("m1_offaxis_front_disp.svg");
    let lim = 4.25;
    let mut ax = plt::chart([-lim, lim, -lim, lim], &fig);
    plt::trimap(&x, &y, &z, &mut ax);

    let r_max = x
        .iter()
        .zip(y.clone())
        .map(|(x, y)| x.hypot(y))
        .fold(f64::NEG_INFINITY, f64::max);
    let (r, o): (Vec<f64>, Vec<f64>) = x
        .iter()
        .zip(y.iter())
        .map(|(x, y)| (x.hypot(*y) / r_max, y.atan2(*x)))
        .unzip();
    println!("max. radius: {}", r_max);
    let n_radial_order = 7;
    let nz = n_radial_order * (n_radial_order + 1) / 2;
    let (j, n, m) = zernike::jnm(n_radial_order);
    let mut zern: Vec<Vec<f64>> = vec![];
    for k in 0..nz as usize {
        zern.push(
            r.iter()
                .zip(o.iter())
                .map(|(r, o)| zernike::zernike(j[k], n[k], m[k], *r, *o))
                .collect(),
        );
    }
    let zgs = zernike::gram_schmidt(
        zern.iter()
            .flatten()
            .cloned()
            .collect::<Vec<f64>>()
            .as_slice(),
        nz as usize,
    );
    let a_zgs = zgs.chunks(n_node).nth(4).unwrap();
    let fig = plt::png_canvas("zgs.png");
    //let fig = plt::canvas("m1_offaxis_front_disp.svg");
    let mut ax = plt::chart([-lim, lim, -lim, lim], &fig);
    plt::trimap(&x, &y, &a_zgs, &mut ax);

    let c: Vec<f64> = zgs
        .chunks(n_node)
        .map(|x| x.iter().zip(z.iter()).fold(0f64, |a, (x, o)| a + x * o))
        .collect();
    let z_e: Vec<_> = zgs
        .chunks(n_node)
        .zip(c)
        .fold(vec![0f64; n_node], |mut a, (x, c)| {
            a.iter_mut().zip(x).for_each(|(a, x)| {
                *a += c * x;
            });
            a
        });

    let fig = plt::png_canvas("z_e.png");
    let mut ax = plt::chart([-lim, lim, -lim, lim], &fig);
    minmax(&z_e);
    plt::trimap(&x, &y, &z_e, &mut ax);
    let z_re: Vec<_> = z.iter().zip(z_e).map(|(x, y)| x - y).collect();

    let file = format!("z_re_{}.svg",nz);
    let fig = plt::canvas(&file);
    let mut ax = plt::chart([-lim, lim, -lim, lim], &fig);
    minmax(&z_re);
    plt::trimap(&x, &y, &z_re, &mut ax);
}
