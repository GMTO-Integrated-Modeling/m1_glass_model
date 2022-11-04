use complot as plt;
use nalgebra as na;
use plotters::prelude::*;
use rayon::prelude::*;
use serde::Deserialize;
use serde_pickle as pkl;
use spade::delaunay::{
    DelaunayTriangulation, DelaunayWalkLocate, FloatDelaunayTriangulation, PositionInTriangulation,
};
use spade::kernels::FloatKernel;
use spade::HasPosition;
use std::{
    env,
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
};

const N_BM_DEFAULT: usize = 27;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Triangulation = DelaunayTriangulation<Surface, FloatKernel, DelaunayWalkLocate>;
type DataShape = (Vec<f64>, (usize, usize));

/// Prints maximum, minimum, mean and standard deviation
pub fn stats(segment: &Segment) {
    let z = &segment.z;
    let minmax = |z: &[f64]| {
        let z_max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z_min = z.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("   - min/max: [{:4.0};{:4.0}]nm", 1e9 * z_min, 1e9 * z_max);
    };
    println!(" - Surface:");
    minmax(z);
    let n = z.len() as f64;
    let z_mean = z.iter().sum::<f64>() / n;
    println!("   - mean: {:4.0}nm", z_mean * 1e9);
    let rms = (z.iter().map(|x| (x - z_mean).powf(2f64)).sum::<f64>() / n).sqrt();
    println!("   - std:  {:4.0}nm", rms * 1e9);
    if let Some(ref bm_f) = segment.bm_forces {
        println!(" - Forces:");
        let pucks: (Vec<f64>, Vec<f64>) =
            pkl::from_reader(File::open("data/pucks.pkl").unwrap(), Default::default()).unwrap();
        let f: Vec<f64> = if segment.tag.starts_with("center") {
            bm_f.iter()
                .zip(pucks.0.iter())
                .map(|(f, p)| *f / (*p))
                .collect()
        } else {
            bm_f.iter()
                .zip(pucks.1.iter())
                .map(|(f, p)| *f / (*p))
                .collect()
        };
        let minmax = |f: &[f64]| {
            let f_max = f.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let f_min = f.iter().cloned().fold(f64::INFINITY, f64::min);
            println!("   - min/max: [{:7.3};{:7.3}]N", f_min, f_max);
        };
        minmax(&f);
        let n = f.len() as f64;
        let f_mean = f.iter().sum::<f64>() / n;
        println!("   - mean: {:7.3}N", f_mean);
        let rms = (f.iter().map(|x| (x - f_mean).powf(2f64)).sum::<f64>() / n).sqrt();
        println!("   - std:  {:7.3}N", rms);
    }
}

/// Segment surface representation
pub struct Surface {
    point: [f64; 2],
    height: f64,
}
impl HasPosition for Surface {
    type Point = [f64; 2];
    fn position(&self) -> [f64; 2] {
        self.point
    }
}

fn polywind(x: f64, y: f64, vx: &[f64], vy: &[f64]) -> i32 {
    let n = vx.len();
    let mut p0 = vx[n - 1];
    let mut p1 = vy[n - 1];
    let mut wind = 0i32;
    for i in 0..n {
        let d0 = vx[i];
        let d1 = vy[i];
        let q = (p0 - x) * (d1 - y) - (p1 - y) * (d0 - x);
        if p1 <= y {
            if d1 > y && q > 0.0 {
                wind += 1;
            }
        } else if d1 <= y && q < 0.0 {
            wind -= 1;
        }
        p0 = d0;
        p1 = d1;
    }
    wind
}
fn truss_shadow(x: f64, y: f64) -> bool {
    let vx = vec![
        -3.011774, -2.446105, -3.011774, -2.799304, -2.33903, -1.566412, -1.640648, -1.65,
        -1.640648, -1.566412, -2.347462, -1.597649, -1.725044, -2.392888, -2.799304,
    ];
    let vy = vec![
        -2.902158, 0., 2.902158, 3.107604, 0.07244, 0.518512, 0.175429, 0., -0.175429, -0.518512,
        -0.067572, -3.865336, -3.810188, -0.427592, -3.107604,
    ];
    (1..4).fold(0, |a, k| {
        let q = geotrans::Quaternion::unit(-120f64.to_radians() * k as f64, geotrans::Vector::k());
        let (vx, vy): (Vec<f64>, Vec<f64>) = vx
            .iter()
            .cloned()
            .zip(vy.iter().cloned())
            .map(|(x, y)| {
                let v = geotrans::Vector::from([x, y, 0.0]);
                let p = geotrans::Quaternion::from(v);
                let pp = &q * p * q.complex_conjugate();
                let u = pp.vector_as_slice();
                (u[0], u[1])
            })
            .unzip();
        a + polywind(x, y, &vx, &vy)
    }) != 0
}
/// Returns the path to the M1 thermal model data
pub fn data_path() -> Result<PathBuf> {
    let path = env::current_dir()?.join("data");
    Ok(path)
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct Data {
    #[serde(rename = "Displacement: Magnitude (m)")]
    pub delta_mag: Option<f64>,
    #[serde(rename = "Displacement[i] (m)")]
    pub delta_x: f64,
    #[serde(rename = "Displacement[j] (m)")]
    pub delta_y: f64,
    #[serde(rename = "Displacement[k] (m)")]
    pub delta_z: f64,
    #[serde(rename = "Temperature (K)")]
    pub temperature: Option<f64>,
    #[serde(rename = "X (m)")]
    pub x: f64,
    #[serde(rename = "Y (m)")]
    pub y: f64,
    #[serde(rename = "Z (m)")]
    pub z: f64,
}

/// M1  bending modes
#[derive(Deserialize)]
pub struct BendingModes {
    /// M1 surface nodes: [x0,y0,x1,y1,...]
    pub nodes: Vec<f64>,
    /// M1 modes
    pub modes: Vec<f64>,
}
impl std::fmt::Display for BendingModes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.nodes.len() / 2;
        write!(
            f,
            "Bending modes: {} modes , {} resolution",
            self.modes.len() / n,
            n,
        )
    }
}
/// M1 segment finite element model
#[derive(Default, Clone)]
pub struct Segment {
    // center or outer
    tag: String,
    /// Segment surface nodes x coordinates
    pub x: Vec<f64>,
    /// Segment surface nodes y coordinates
    pub y: Vec<f64>,
    /// Segment surface nodes z coordinates
    pub z: Vec<f64>,
    /// Surface temperature field
    pub temperature: Option<Vec<f64>>,
    // data repository
    model_dir: String,
    /// Bending modes forces
    pub bm_forces: Option<Vec<f64>>,
    n_bm: usize,
}
impl Segment {
    /// Builds a new model
    ///
    /// Loads surface deformation from CSV file "\<tag\>_disp.csv" in `model_dir` directory  for either `tag`="center" or `tag`="outer"
    pub fn new(model_dir: &str, tag: &str) -> Result<Self> {
        let datapath = data_path()?
            .join(model_dir)
            .join(format!("{}_disp.csv", tag));
        println!("Reading {:?}...", datapath);
        let mut rdr = csv::Reader::from_path(datapath)?;
        let mut data: Vec<Data> = vec![];
        for d in rdr.deserialize() {
            data.push(d?);
        }
        let n_node = data.len();
        println!("{} records!", n_node);
        let (x, yz): (Vec<f64>, Vec<(f64, f64)>) = data
            .iter()
            .filter(|d| d.x.hypot(d.y) <= 0.5 * 8.365)
            .map(|d| (d.x + d.delta_x, (d.y + d.delta_y, d.delta_z)))
            .unzip();
        let (y, z): (Vec<f64>, Vec<f64>) = yz.into_iter().unzip();

        let minmax = |z: &[f64]| {
            let z_max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z_min = z.iter().cloned().fold(f64::INFINITY, f64::min);
            println!("min/max: [{:6.3e};{:6.3e}]", z_min, z_max);
        };
        let temperature: Vec<_> = data
            .iter()
            .filter(|d| d.x.hypot(d.y) <= 0.5 * 8.365)
            .filter_map(|d| d.temperature)
            .collect();

        Ok(if temperature.is_empty() {
            Self {
                tag: tag.to_owned(),
                x,
                y,
                z,
                temperature: None,
                model_dir: model_dir.to_owned(),
                n_bm: N_BM_DEFAULT,
                ..Default::default()
            }
        } else {
            //println!("Surface [nm]:");
            //minmax(&z);
            println!("Temperature [K]:");
            minmax(&temperature);
            Self {
                tag: tag.to_owned(),
                x,
                y,
                z,
                temperature: Some(temperature),
                model_dir: model_dir.to_owned(),
                n_bm: N_BM_DEFAULT,
                ..Default::default()
            }
        })
    }
    pub fn new_with(model_dir: &str, tag: &str, n_bm: usize) -> Result<Self> {
        Self::new(model_dir, tag).map(|mut x| {
            x.n_bm = n_bm;
            x
        })
    }
    /// Dumps data into a pickle file
    ///
    /// Pickle either [Info::Temperature] or [Info::Surface]
    pub fn to_pkl<P>(&self, path: P, info: Info) -> Result<()>
    where
        P: AsRef<std::path::Path> + std::fmt::Display + Copy,
    {
        use serde::Serialize;
        #[derive(Serialize)]
        struct Field {
            nodes: Vec<f64>, //x0,y0,x1,y1,...
            field: Vec<f64>,
        }
        let nodes: Vec<_> = self
            .x
            .iter()
            .zip(self.y.iter())
            .flat_map(|(x, y)| vec![*x, *y])
            .collect();
        match info {
            Info::Temperature => {
                if let Some(temperature) = &self.temperature {
                    let field = Field {
                        nodes,
                        field: temperature.clone(),
                    };
                    let file = File::create(data_path()?.join(&self.model_dir).join(path))?;
                    let mut writer = BufWriter::with_capacity(100_000, file);
                    pkl::to_writer(&mut writer, &field, Default::default())?
                }
            }
            Info::Surface => {
                let field = Field {
                    nodes,
                    field: self.z.clone(),
                };
                let file = File::create(data_path()?.join(&self.model_dir).join(path))?;
                let mut writer = BufWriter::with_capacity(100_000, file);
                pkl::to_writer(&mut writer, &field, Default::default())?
            }
            _ => unimplemented!("only temperature and surface can be pickled!"),
        }
        Ok(())
    }
    /// Filters surface deformation
    ///
    /// Least square fits and removes piston, tip, tilt and bending modes
    pub fn filter_surface(
        &self,
        surface: &[f64],
        bending: Option<BendingModes>,
        extra_bm: Option<String>,
        only_bm: Option<String>,
        rbms: Option<Vec<f64>>,
        force_matrix: Option<DataShape>,
    ) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
        let n_node = self.x.len();
        let mut projection_columns = match rbms {
            Some(ref rbms) => {
                println!("## Filtering M1 rigid body motions ##");
                rbms.chunks(n_node)
                    .map(|x| na::DVector::from_column_slice(x))
                    .chain(Some(na::DVector::from_element(n_node, 1f64)))
                    .collect()
            }
            None => {
                println!("## Filtering piston, tip and tilt ##");
                let z0 = na::DVector::from_element(n_node, 1f64);
                let z1 = na::DVector::from_column_slice(&self.x);
                let z2 = na::DVector::from_column_slice(&self.y);
                vec![z0, z1, z2]
            }
        };
        let n_rbm = projection_columns.len();
        // println!("## # of modes: {} ##", n_rbm);
        if let Some(ref x) = bending {
            let n_bm_max = 2 * x.modes.len() / x.nodes.len() - 3;
            let n_bm = env::var("N_BM")
                .map_or(self.n_bm, |v| v.parse::<usize>().unwrap_or(self.n_bm))
                .min(n_bm_max);
            match extra_bm {
                Some(extras) => {
                    let bm_idx: Vec<_> = extras
                        .split(',')
                        .filter_map(|x| x.parse::<usize>().ok())
                        .collect();
                    println!(
                        "## Filtering first {} bending modes and modes {:?} from {} ##",
                        n_bm, bm_idx, self.tag
                    );
                    let extras = bm_idx
                        .into_iter()
                        .filter_map(|k| x.modes.chunks(n_node).nth(k - 1));
                    let bm = x
                        .modes
                        .chunks(n_node)
                        .take(n_bm)
                        .chain(extras)
                        .map(|x| na::DVector::from_column_slice(x));
                    projection_columns.extend(bm);
                }
                None => {
                    println!("## Filtering {} bending modes from {} ##", n_bm, self.tag);
                    let bm = x
                        .modes
                        .chunks(n_node)
                        .take(n_bm)
                        .map(|x| na::DVector::from_column_slice(x));
                    projection_columns.extend(bm);
                }
            };
        };
        let projection = na::DMatrix::from_columns(&projection_columns);
        //      println!("projection: {:?}", projection.shape());
        let projection_svd = projection.svd(true, true);
        let sing_values = projection_svd.singular_values;
        let cond = sing_values.max() / sing_values.min();
        log::debug!("Projection condition #: {:.6e}", cond);
        let u = projection_svd
            .u
            .as_ref()
            .ok_or("Failed getting left eigen modes")?;
        let s = na::DVector::from_column_slice(surface);
        let mut w = &u.transpose() * &s;
        if let Some(onlys) = only_bm {
            let bm_idx: Vec<_> = onlys
                .split(',')
                .filter_map(|x| x.parse::<usize>().ok())
                .collect();
            w.iter_mut()
                .skip(n_rbm)
                .enumerate()
                .filter(|(i, _)| !bm_idx.contains(&(*i + 1)))
                .for_each(|(_, w)| *w = 0f64);
        }
        let filename = format!("{}_coefs.pkl", self.tag);
        let ww = w.as_slice().to_owned();
        pkl::to_writer(
            &mut File::create(filename).unwrap(),
            &ww,
            Default::default(),
        )
        .unwrap();

        let forces = if let (Some(ref fm), Some(ref bm)) = (force_matrix, bending) {
            Some(self.forces(&w.as_slice()[13..], bm, fm))
        } else {
            None
        };

        let q = s - u * w;
        match rbms {
            Some(rbms) if self.tag == "outer_on-bm" => {
                let pupil = rbms.chunks(n_node).fold(vec![1f64; n_node], |mut s, x| {
                    x.iter().zip(s.iter_mut()).for_each(|(x, s)| {
                        if *x == 0f64 {
                            *s = 0f64
                        }
                    });
                    s
                });
                Ok((
                    q.as_slice()
                        .iter()
                        .zip(pupil.iter())
                        .map(|(x, p)| x * p)
                        .collect::<Vec<f64>>(),
                    forces,
                ))
            }
            _ => Ok((q.as_slice().to_vec(), forces)),
        }
    }
    /// Derives the actuator forces from the bending modes
    pub fn forces(&self, b: &[f64], _bending: &BendingModes, force_matrix: &DataShape) -> Vec<f64> {
        let b2f = {
            // let n_mode = bending.modes.len() / (bending.nodes.len() / 2);
            let fm = na::DMatrix::from_column_slice(
                force_matrix.1 .0,
                force_matrix.1 .1,
                &force_matrix.0,
            );
            let fm_svd = fm.svd(true, true);
            fm_svd.v_t.unwrap().transpose()
                * na::DMatrix::from_diagonal(&fm_svd.singular_values.map(|x| x.recip()))
        };
        {
            let n_b = b.len();
            b2f.columns(0, n_b) * na::DVector::from_column_slice(b)
        }
        .as_slice()
        .to_vec()
    }
    /// Delaunay triangulates the surface
    pub fn triangulate(&self, z: &[f64]) -> Triangulation {
        let mut tri = FloatDelaunayTriangulation::with_walk_locate();
        self.x
            .iter()
            .zip(self.y.iter())
            .zip(z)
            .for_each(|((x, y), z)| {
                tri.insert(Surface {
                    point: [*x, *y],
                    height: *z,
                });
            });
        tri
    }
    /// Barycentric interpolation of the surface
    ///
    /// Interpolant coordinates are given as [chunks](std::slice::chunks) of size 2: [x,y]
    pub fn interpolate<'a>(
        &self,
        xyi: impl Iterator<Item = &'a [f64]>,
        tri: &Triangulation,
    ) -> Result<Vec<f64>> {
        xyi.map(|xyi| {
            tri.barycentric_interpolation(&[xyi[0], xyi[1]], |p| p.height)
                .ok_or_else(|| "Failed interpolation".into())
        })
        .collect()
    }
    /// Geometric transformation of the surface coordinates system
    ///
    /// The surface coordinates are transformed in M1 segment coordinates
    pub fn to_local(mut self, sid: usize) -> Self {
        self.x.iter_mut().zip(self.y.iter_mut()).for_each(|(x, y)| {
            let v = geotrans::oss_to_any_m1(sid, [*x, *y, 0.]);
            *x = v[0];
            *y = v[1];
        });
        self
    }
}
/// Information data type
#[derive(Clone)]
pub enum Info {
    Surface,
    ResidualSurface,
    Temperature,
}
impl Segment {
    pub fn show(&self, info: Info) -> Result<&Self> {
        let datapath = data_path()?.join(&self.model_dir);
        match info {
            Info::Surface => {
                let filename = datapath.join(format!("actuator_heat_{}_surface.png", self.tag));
                let l = 4.2f64;
                let figure = BitMapBackend::new(&filename, (1024, 1024)).into_drawing_area();
                let mut axis = ChartBuilder::on(&figure)
                    .set_label_area_size(LabelAreaPosition::Left, 50)
                    .set_label_area_size(LabelAreaPosition::Bottom, 50)
                    .margin_top(50)
                    .margin_right(50)
                    .build_cartesian_2d(-l..l, -l..l)
                    .unwrap();
                plt::tri::trimap(&self.x, &self.y, &self.z, &mut axis);
            }
            Info::ResidualSurface => {
                let filename =
                    datapath.join(format!("actuator_heat_{}_residual-surface.png", self.tag));
                let (ze, _) = self.filter_surface(&self.z, None, None, None, None, None)?;
                let l = 4.2f64;
                let figure = BitMapBackend::new(&filename, (1024, 1024)).into_drawing_area();
                let mut axis = ChartBuilder::on(&figure)
                    .set_label_area_size(LabelAreaPosition::Left, 50)
                    .set_label_area_size(LabelAreaPosition::Bottom, 50)
                    .margin_top(50)
                    .margin_right(50)
                    .build_cartesian_2d(-l..l, -l..l)
                    .unwrap();
                plt::tri::trimap(&self.x, &self.y, &ze, &mut axis);
            }
            Info::Temperature => {
                if let Some(temperature) = &self.temperature {
                    let filename =
                        datapath.join(format!("actuator_heat_{}_temperature.png", self.tag));
                    let l = 4.2f64;
                    let figure = BitMapBackend::new(&filename, (1024, 1024)).into_drawing_area();
                    let mut axis = ChartBuilder::on(&figure)
                        .set_label_area_size(LabelAreaPosition::Left, 50)
                        .set_label_area_size(LabelAreaPosition::Bottom, 50)
                        .margin_top(50)
                        .margin_right(50)
                        .build_cartesian_2d(-l..l, -l..l)
                        .unwrap();
                    plt::tri::trimap(&self.x, &self.y, &temperature, &mut axis);
                    let legend_plot = figure.clone();
                    //.shrink((n_grid as u32 - 50, 40), (50, n_grid as u32 - 80));
                    let n_grid = 800;
                    let cells_max = temperature
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    let cells_min = temperature.iter().cloned().fold(f64::INFINITY, f64::min);
                    let mut legend = ChartBuilder::on(&legend_plot)
                        .set_label_area_size(LabelAreaPosition::Right, 50)
                        .margin_top(50)
                        .margin_bottom(50)
                        .build_cartesian_2d(0..n_grid, cells_min..cells_max)
                        .unwrap();
                    legend
                        .configure_mesh()
                        .disable_x_mesh()
                        .disable_y_mesh()
                        .axis_desc_style(
                            TextStyle::from(("sans-serif", 14).into_font()).color(&WHITE),
                        )
                        .label_style(TextStyle::from(("sans-serif", 12).into_font()).color(&WHITE))
                        .y_desc("Temperature [K]")
                        .draw()
                        .unwrap();
                    let legend_area = legend.plotting_area();
                    for i in 0..20 {
                        let x = n_grid - 20 + i;
                        for j in 0..n_grid {
                            let u = j as f64 / n_grid as f64;
                            let y = u * (cells_max - cells_min) + cells_min;
                            legend_area
                                .draw_pixel((x, y), &HSLColor(0.5 * u, 0.5, 0.4))
                                .unwrap();
                        }
                    }
                }
            }
        };
        Ok(self)
    }
}
/// Segment type variants
pub enum Segments<T> {
    /// Center segment
    Center(T),
    /// Outer segment
    Outer(T),
}
impl Segments<Segment> {
    /// Dumps segment data into a pickle file
    ///
    /// Pickle either [Info::Temperature] or [Info::Surface] into `center.pkl` or `outer.pkl`
    pub fn to_pkl(&self, info: Info) -> Result<&Self> {
        match self {
            Segments::Center(segment) => segment.to_pkl("center.pkl", info)?,
            Segments::Outer(segment) => segment.to_pkl("outer.pkl", info)?,
        }
        Ok(self)
    }
    pub fn show(&self, info: Info) -> Result<&Self> {
        match self {
            Segments::Center(segment) => segment.show(info)?,
            Segments::Outer(segment) => segment.show(info)?,
        };
        Ok(self)
    }
    /// Loads the bending modes
    ///
    /// The bending modes are expected to be in the data directory in the files `data/bending_modes_CS.pkl` or `data/bending_modes_OA.pkl`
    pub fn bending_modes(&self) -> Result<BendingModes> {
        let filename = match self {
            Segments::Center(_) => "data/bending_modes_CS.pkl",
            Segments::Outer(_) => "data/bending_modes_OA.pkl",
        };
        let bm_file = File::open(filename)?;
        let rdr = BufReader::with_capacity(100_000, bm_file);
        let bending: BendingModes = pkl::from_reader(rdr, Default::default())?;
        Ok(bending)
    }
    /// Loads the bending modes force matrix
    ///
    /// The bending modes force matrices are expected to be in the data directory in the files `data/bending_modes_CS_af.pkl` or `data/bending_modes_OA_af.pkl`
    pub fn bending_modes_force_matrix(&self) -> Result<DataShape> {
        let filename = match self {
            Segments::Center(_) => "data/bending_modes_CS_af.pkl",
            Segments::Outer(_) => "data/bending_modes_OA_af.pkl",
        };
        let bm_file = File::open(filename)?;
        let rdr = BufReader::with_capacity(100_000, bm_file);
        let force_matrix: DataShape = pkl::from_reader(rdr, Default::default())?;
        Ok(force_matrix)
    }
    /// Resamples the segment surface onto bending modes nodes coordinates
    pub fn resample_on_bending_modes(&self) -> Result<Segments<Segment>> {
        match self {
            Segments::Center(segment) => {
                let tri = segment.triangulate(&segment.z);
                self.bending_modes().and_then(|bm| {
                    let z = segment.interpolate(bm.nodes.chunks(2), &tri)?;
                    let (x, y): (Vec<f64>, Vec<f64>) =
                        bm.nodes.chunks(2).map(|x| (x[0], x[1])).unzip();
                    Ok(Segments::Center(Segment {
                        tag: "center_on-bm".to_owned(),
                        x,
                        y,
                        z,
                        ..segment.clone()
                    }))
                })
            }
            Segments::Outer(segment) => {
                let tri = segment.triangulate(&segment.z);
                self.bending_modes().and_then(|bm| {
                    let z = segment.interpolate(bm.nodes.chunks(2), &tri)?;
                    let (x, y): (Vec<f64>, Vec<f64>) =
                        bm.nodes.chunks(2).map(|x| (x[0], x[1])).unzip();
                    Ok(Segments::Outer(Segment {
                        tag: "outer_on-bm".to_owned(),
                        x,
                        y,
                        z,
                        ..segment.clone()
                    }))
                })
            }
        }
    }
    /// Filters surface deformation
    ///
    /// Least square fits and removes piston, tip, tilt and bending modes
    pub fn filter_surface(self) -> Result<Segments<Segment>> {
        let bm = self.bending_modes();
        let fm = self.bending_modes_force_matrix();
        match self {
            Segments::Center(segment) => {
                let rbms: Option<Vec<f64>> = env::var("RBMS")
                    .and({
                        let file = File::open("data/RBMS_CS.pkl")?;
                        Ok(pkl::from_reader(&file, Default::default())?)
                    })
                    .ok();
                let extra_bm = env::var("EXTRA_BM_CENTER").ok();
                let only_bm = env::var("ONLY_BM_CENTER").ok();
                let (z, f) = segment.filter_surface(
                    &segment.z,
                    bm.ok(),
                    extra_bm,
                    only_bm,
                    rbms,
                    fm.ok(),
                )?;
                Ok(Segments::Center(Segment {
                    tag: "center_on-bm_filtered".to_owned(),
                    z,
                    bm_forces: f,
                    ..segment
                }))
            }
            Segments::Outer(segment) => {
                let rbms: Option<Vec<f64>> = env::var("RBMS")
                    .and({
                        let file = File::open("data/RBMS_OA.pkl")?;
                        Ok(pkl::from_reader(&file, Default::default())?)
                    })
                    .ok();
                let extra_bm = env::var("EXTRA_BM_OUTER").ok();
                let only_bm = env::var("ONLY_BM_OUTER").ok();
                let (z, f) = segment.filter_surface(
                    &segment.z,
                    bm.ok(),
                    extra_bm,
                    only_bm,
                    rbms,
                    fm.ok(),
                )?;
                Ok(Segments::Outer(Segment {
                    tag: "outer_on-bm_filtered".to_owned(),
                    z,
                    bm_forces: f,
                    ..segment
                }))
            }
        }
    }
    /// Prints segment surface statistics
    pub fn stats(&self) {
        match self {
            Segments::Center(segment) => stats(&segment),
            Segments::Outer(segment) => stats(&segment),
        };
    }
    /// Clones segments and transforms the surface into the segment coordinates system
    pub fn clone_to(&mut self, sid: usize) -> Segment {
        match self {
            Segments::Center(segment) => segment.clone().to_local(sid),
            Segments::Outer(segment) => segment.clone().to_local(sid),
        }
    }
}
/// M1 seven segments mirror assembly
pub struct Mirror {
    center: Segments<Segment>,
    outer: Segments<Segment>,
    segments: Option<Vec<Segment>>,
    case: String,
}
impl Mirror {
    /// Creates a new mirror from a given case path
    pub fn new(case: &str) -> Result<Self> {
        Ok(Self {
            outer: Segments::Outer(Segment::new(case, "outer")?),
            center: Segments::Center(Segment::new(case, "center")?),
            segments: None,
            case: case.to_owned(),
        })
    }
    pub fn new_with(case: &str, n_bm: usize) -> Result<Self> {
        Ok(Self {
            outer: Segments::Outer(Segment::new_with(case, "outer", n_bm)?),
            center: Segments::Center(Segment::new_with(case, "center", n_bm)?),
            segments: None,
            case: case.to_owned(),
        })
    }
    /// Dumps segments data to pickle file
    pub fn to_pkl(&self, info: Info) -> Result<&Self> {
        self.center.to_pkl(info.clone())?;
        self.outer.to_pkl(info)?;
        Ok(self)
    }
    pub fn show(&self, info: Info) -> Result<&Self> {
        self.center.show(info.clone())?;
        self.outer.show(info)?;
        Ok(self)
    }
    pub fn show_whole(&self) -> Result<()> {
        let l = 13f64;
        let filename = data_path()?
            .join(&self.case)
            .join("actuator_heat_whole.png");
        let figure = BitMapBackend::new(&filename, (1024, 1024)).into_drawing_area();
        let mut axis = ChartBuilder::on(&figure)
            .build_cartesian_2d(-l..l, -l..l)
            .unwrap();
        self.segments
            .as_ref()
            .unwrap()
            .iter()
            .for_each(|s| plt::tri::trimap(&s.x, &s.y, &s.z, &mut axis));
        Ok(())
    }
    /// Resamples the segments surface onto bending modes nodes coordinates
    pub fn resample_on_bending_modes(self) -> Result<Mirror> {
        Ok(Mirror {
            center: self.center.resample_on_bending_modes()?,
            outer: self.outer.resample_on_bending_modes()?,
            segments: None,
            case: self.case,
        })
    }
    /// Filters segments surface deformation
    ///
    /// Least square fits and removes piston, tip, tilt and bending modes
    pub fn filtered(self) -> Result<Mirror> {
        Ok(Mirror {
            center: self.center.filter_surface()?,
            outer: self.outer.filter_surface()?,
            segments: None,
            case: self.case,
        })
    }
    /// Prints segment surface statistics
    pub fn stats(&self) -> &Self {
        println!("Center");
        self.center.stats();
        println!("Outer");
        self.outer.stats();
        self
    }
    /// Geometric transformation of segments surface coordinates
    ///
    /// The surface coordinates are transformed in M1 segment coordinates
    pub fn to_local(&mut self) -> &Self {
        let mut segments: Vec<_> = (1..=6)
            .into_iter()
            .map(|sid| self.outer.clone_to(sid))
            .collect();
        segments.push(self.center.clone_to(7));
        self.segments = Some(segments);
        self
    }
    /// Interpolates segments surface deformation on a regular mesh
    ///
    /// Interpolates the seven segments on a regular `n_grid`X`n_grid` mesh of size `length`[m] X `length`[m]
    /// Outputs are the pupil (1 inside segments, 0 outside) and the wavefront as the interpolated surfaces X2
    pub fn gridding(&self, length: f64, n_grid: usize) -> (Vec<f64>, Vec<f64>) {
        let tri: Vec<Triangulation> = self
            .segments
            .as_ref()
            .unwrap()
            .par_iter()
            .map(|segment| {
                let mut tri = DelaunayTriangulation::with_walk_locate();
                segment
                    .x
                    .iter()
                    .zip(segment.y.iter())
                    .zip(segment.z.iter())
                    .for_each(|((x, y), z)| {
                        tri.insert(Surface {
                            point: [*x, *y],
                            height: *z,
                        });
                    });
                tri
            })
            .collect();
        let d = length / (n_grid - 1) as f64;
        (0..n_grid * n_grid)
            .into_par_iter()
            .map(|k| {
                let x = (k / n_grid) as f64 * d - 0.5 * length;
                let y = (k % n_grid) as f64 * d - 0.5 * length;
                if x.hypot(y) < 1.75 || truss_shadow(x, y) {
                    (0f64, 0f64)
                } else {
                    tri.iter()
                        .find_map(|t| {
                            let p = [x, y];
                            match t.locate(&p) {
                                PositionInTriangulation::OutsideConvexHull(_) => None,
                                _ => t
                                    .barycentric_interpolation(&p, |p| p.height)
                                    .map(|x| (1f64, 2f64 * x)),
                            }
                        })
                        .or(Some((0f64, 0f64)))
                        .unwrap()
                }
            })
            .unzip()
    }
}
