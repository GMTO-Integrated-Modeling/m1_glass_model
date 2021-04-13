# CFD M1 Segment Surface deformation

Convert M1 thermal surface figure into wavefront error

## Installation

First, install [Rust](https://www.rust-lang.org/tools/install), then at a terminal install the model with

```
git clone https://github.com/GMTO/m1_glass_model.git
cd m1_glass_model
```

To run the model, you need to also download the data in the `m1_glass_model` folder:
```
wget https://s3-us-west-2.amazonaws.com/gmto.modeling/m1_glass_model.tgz .
tar -xzvf m1_glass_model.tgz
```

## Usage

To get a description of the inputs to the model: `cargo run --release -- --help` and to run a simulation of a given case use: `cargo run --release -- --case <case_name>`, the input data will read from the case folder and the output pictures will be written to the same folder.

There are 7 cases to choose from:
 - case1 : Set the initial glass temperature at ambient. Simulate for 4h while ambient decreases at a rate of -0.2K/h. Assume that the actuators dissipate heat in the lower plenum and all of it is distributed in the nearest nozzles, increasing their nominal temperature above Tn. The HTC values used for the core surfaces correspond to less accurate older unsteady simulations.
 - case2 : Set the initial glass temperature at ambient. Simulate for 4h while ambient decreases at a rate of -0.2K/h. Assume that the actuators dissipate heat in the upper plenum which increases the nominal temperature above Tu near the back mirror surface. The HTC values used for the core surfaces correspond to less accurate older unsteady simulations.
 - tm1optimal : Set the initial glass temperature at ambient -0.05K. Simulate for 4h while ambient decreases at a rate of -0.1K/h. Assume that the actuators dissipate heat in the upper plenum. Assume that the HTC value for the core sides is radially maintained at the center core value of 6.5 W/m2/K, and that the mean HTC at the back side of the mirror has the same value. The value on the top core surface is also maintained radially. Even though Table 1 showed that the latter is possible by radially increasing the nozzle height, the former is really not feasible (some radial degradation is inevitable), hence the word “optimal”.
 - tm1cs (constant sides): Set the initial glass temperature at ambient -0.05K. Simulate for 4h while ambient decreases at a rate of -0.1K/h. Assume that the actuators dissipate heat in the upper plenum. Assume that the HTC value for the core sides is radially maintained, but at a reduced center core value of 2.8 W/m2/K. The mean HTC at the back side of the mirror has the same value as in the “optimal” case. The value on the top core surface is also maintained radially. This was to test the effect of the absolute core side HTC value alone.
 tm1radial : Set the initial glass temperature at ambient -0.05K. Simulate for 4h while ambient decreases at a rate of -0.1K/h. Assume that the actuators dissipate heat in the upper plenum. Assume that the center core HTC value for the core sides is 6.5 W/m2/K but it degrades radially as estimated. The mean HTC at the back side of the mirror has the same value as in the “optimal” case. The value on the top core surface also degrades radially. This was to test the effect of radial degradation of the HTC values alone.
- real_radial_nozzle: Set the initial glass temperature at ambient -0.05K. Simulate for 4h while ambient decreases at a rate of -0.1K/h. Assume that the actuators dissipate heat in the lower plenum, affecting the nozzle temperatures. Assume the nozzle heights increase according to the function fNH = 0.025+0.015r2 (m) so that the distance between the nozzle exit and the core top is radially maintained. Curve-fit HTC values appropriately. Update mirror back and side HTC values from the latest simulations.
 - real2 : Set the initial glass temperature at ambient -0.05K. Simulate for 4h while ambient decreases at a rate of -0.1K/h. Assume that the actuators dissipate heat in the upper plenum. Assume the nozzle heights increase according to the function fNH = 0.025+0.015r2 (m) so that the distance between the nozzle exit and the core top is radially maintained. Curve-fit HTC values appropriately. Update mirror back and side HTC values from the latest simulations
