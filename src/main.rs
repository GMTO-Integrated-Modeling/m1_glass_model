use glass::{Info, Mirror};
use gmt_kpp::KPP;
use plotters::prelude::*;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
/// Convert M1 thermal surface figure into wavefront error
///
/// Load the surface thermal deformation for the outer and center segment from 2 files:
/// "outer_disp.csv" and "center_disp.csv", respectively, according to the selected case.
/// Per default, piston, tilt-tilt and the fist 27 bending modes are fitted to and
/// removed from the surfaces. The number of bending modes is altered with the environment
/// variable N_BM; if  N_BM is larger than the total number of bending modes,
/// it is clipped to 162 for the outer segment and 151 for the center segment.
/// Bending modes can also be selected one-by-one with the environment variables
/// EXTRA_BM_OUTER and EXTRA_BM_CENTER for the outer and center segments, respectively.
/// Both variables are set to a comma separated list.
/// On Linux, in the bash shell, the variable are set with:
///
/// >>> export N_BM=27
///
/// >>> export EXTRA_BM_OUTER=46
///
/// >>> export EXTRA_BM_CENTER=40
#[structopt(name = "glass", about = "CFD M1 segment surface deformation")]
struct Opt {
    /// Case folder name
    ///
    /// The case folder is expected to be in the directory data from the current working directory
    #[structopt(short, long)]
    case: String,
}
#[derive(Debug)]
enum PSSN {
    V(f64),
    H(f64),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();
    let case = opt.case;
    let mirror = Mirror::new(&case)?;
    //mirror.to_pkl(Info::Temperature)?;
    mirror.stats();
    mirror.show(Info::Temperature)?;
    mirror.show(Info::Surface)?;
    mirror.show(Info::ResidualSurface)?;
    let mut mirror = mirror.resample_on_bending_modes()?.filtered()?;
    mirror.stats().show(Info::Surface)?;
    mirror.to_local().show_whole()?;
    println!("Gridding ...");
    let length = 25.5f64;
    let n_grid = 769;
    let (pupil, wavefront) = mirror.gridding(length, n_grid);
    let pssn = [
        PSSN::V({
            let mut pssn = KPP::new().wavelength(500e-9).pssn(length, n_grid, &pupil);
            pssn.estimate(&pupil, Some(&wavefront))
        }),
        PSSN::H({
            let mut pssn = KPP::new().wavelength(1.65e-6).pssn(length, n_grid, &pupil);
            pssn.estimate(&pupil, Some(&wavefront))
        }),
    ];
    println!("PSSN:{:.4?}", pssn);

    let filename = glass::data_path()?.join(&case).join("wavefront.png");
    let plot = BitMapBackend::new(&filename, (n_grid as u32 + 100, n_grid as u32 + 100))
        .into_drawing_area();
    plot.fill(&BLACK).unwrap();
    let l = length / 2.;
    let mut chart = ChartBuilder::on(&plot)
        .set_label_area_size(LabelAreaPosition::Left, 50)
        .set_label_area_size(LabelAreaPosition::Bottom, 50)
        .margin_top(50)
        .margin_right(50)
        .build_cartesian_2d(-l..l, -l..l)
        .unwrap();
    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()
        .unwrap();
    let cells_max = wavefront.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let cells_min = wavefront.iter().cloned().fold(f64::INFINITY, f64::min);
    let unit_wavefront: Vec<f64> = wavefront
        .iter()
        .map(|p| (p - cells_min) / (cells_max - cells_min))
        .collect();
    let plotting_area = chart.plotting_area();
    let d = length / (n_grid - 1) as f64;
    for i in 0..n_grid {
        let x = i as f64 * d - 0.5 * length;
        for j in 0..n_grid {
            let y = j as f64 * d - 0.5 * length;
            let ij = i * n_grid + j;
            if pupil[ij] != 0.0 {
                plotting_area
                    .draw_pixel((x, y), &HSLColor(0.5 * unit_wavefront[ij], 0.5, 0.4))
                    .unwrap();
            }
        }
    }
    Ok(())
}
