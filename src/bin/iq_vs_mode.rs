use std::fs::File;

use glass::Mirror;
use gmt_kpp::KPP;
use rayon::prelude::*;
use serde::Serialize;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "M1 Glass Model",
    about = "Estimate image quality as a function of the number of bending modes up to 42"
)]
struct Opt {
    /// Case folder name
    ///
    /// The case folder is expected to be in the directory data from the current working directory
    #[structopt(short, long)]
    case: String,
}

#[derive(Debug, Serialize)]
struct PSSN {
    v_band: f64,
    h_band: f64,
}
#[derive(Debug, Serialize)]
struct ImageQuality {
    n_mode: usize,
    wfe_rms: f64,
    pssn: PSSN,
}

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();
    let case = opt.case;

    let iq: Vec<_> = (3..=42)
        .into_par_iter()
        .map(|n_mode| {
            // env::set_var("N_BM", format!("{n_mode}"));
            let mirror = Mirror::new_with(&case, n_mode).expect(&format!("Case {case} not found!"));
            let mut mirror = mirror
                .resample_on_bending_modes()
                .unwrap()
                .filtered()
                .unwrap();
            mirror.to_local();
            let length = 25.5f64;
            let n_grid = 769;
            let (pupil, wavefront) = mirror.gridding(length, n_grid);
            let w: Vec<_> = pupil
                .iter()
                .zip(wavefront.iter())
                .filter_map(|(p, w)| if *p > 0f64 { Some(w) } else { None })
                .collect();
            let n = w.len() as f64;
            let mean_w = w.iter().cloned().sum::<f64>() / n;
            let wfe_rms = (w
                .into_iter()
                .map(|x| {
                    let y = x - mean_w;
                    y * y
                })
                .sum::<f64>()
                / n)
                .sqrt()
                * 1e9;
            let pssn = PSSN {
                v_band: {
                    let mut pssn = KPP::new().wavelength(500e-9).pssn(length, n_grid, &pupil);
                    pssn.estimate(&pupil, Some(&wavefront))
                },
                h_band: {
                    let mut pssn = KPP::new().wavelength(1.65e-6).pssn(length, n_grid, &pupil);
                    pssn.estimate(&pupil, Some(&wavefront))
                },
            };
            ImageQuality {
                n_mode,
                wfe_rms,
                pssn,
            }
        })
        .collect();
    serde_pickle::to_writer(
        &mut File::create("image_quality.pkl")?,
        &iq,
        Default::default(),
    )?;
    Ok(())
}
