use complot as plt;
use glass::{Info, Mirror};
use gmt_kpp::KPP;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
/// Convert M1 thermal surface figure into wavefront error
///
/// Load the surface thermal deformation for the outer and center segment from 2 files:
/// "outer_disp.csv" and "center_disp.csv", respectively, according to the selected case.
/// Per default, piston, tilt-tilt and the fist 27 bending modes are fitted to and
/// removed from the surfaces.
/// Instead of piston and tip-tilt, M1 and M2 rigid body motions can be used by setting
/// the environment variable RBMS=1.
/// The number of bending modes is altered with the environment
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
#[structopt(name = "M1 Glass Model", about = "CFD M1 segment surface deformation")]
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
    env_logger::init();

    let opt = Opt::from_args();
    let case = opt.case;
    let mirror = Mirror::new(&case)?;
    //mirror.to_pkl(Info::Temperature)?;
    mirror.stats();
    mirror.show(Info::Temperature)?;
    mirror.show(Info::Surface)?;
    //mirror.show(Info::ResidualSurface)?;

    let mut mirror = mirror.resample_on_bending_modes()?.filtered()?;
    mirror.stats().show(Info::Surface)?;
    mirror.to_local().show_whole()?;
    println!("Gridding ...");
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
    let std_w = (w
        .into_iter()
        .map(|x| {
            let y = x - mean_w;
            y * y
        })
        .sum::<f64>()
        / n)
        .sqrt()
        * 1e9;
    println!("WFE RMS: {:.3}nm", std_w);
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
    println!("PSSN:{:.6?}", pssn);

    let filename = glass::data_path()?.join(&case).join("wavefront.png");
    let _: plt::Heatmap<f64> = (
        (
            wavefront
                .iter()
                .map(|x| x * 1e9)
                .collect::<Vec<f64>>()
                .as_slice(),
            (n_grid, n_grid),
        ),
        plt::Config::new()
            .filename(filename.to_str().unwrap())
            .xaxis(plt::XAxis::new().label("Wavefront error [nm]")),
    )
        .into();
    Ok(())
}
