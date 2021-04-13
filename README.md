# CFD M1 Segment Surface deformation

Convert M1 thermal surface figure into wavefront error

## Installation

First, install [Rust](https://www.rust-lang.org/tools/install), then at a terminal install the model with

`cargo install --git https://github.com/GMTO/m1_glass_model.git --branch main` 

and finally download the [data](https://s3-us-west-2.amazonaws.com/gmto.modeling/m1_glass_model.tgz).

## Usage

At a terminal enter: 

`tar -xzvf m1_glass_model.tgz`

then, to get a description of the inputs to the model: `glass --help` and to run a simulation of a give case use: `glass --case <case_name>`
