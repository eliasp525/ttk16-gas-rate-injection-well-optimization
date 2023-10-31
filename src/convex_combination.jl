using JuMP
using GLPK
using DataFrames
using CSV
using LinearAlgebra
using Plots

### Data

const DATA_DIR = "/home/elias/Documents/ntnu/fordypning/julia/ttk16/oving4/data";

N = 8; # number of wells
Wells = 1:N;
Datapoints = Containers.DenseAxisArray([6, 8, 9, 5, 5, 8, 7, 6], Wells); # Number of datapoints for each well
DatapointsWell = fill(Int[],1,N);  # Index sets for datapoints for each well
for n = Wells
    DatapointsWell[n] = 1:Datapoints[n];
end

Qinj_max = 500; # Max gas injection
Qliq_max = 5500; # Max liquid production
Qgas_max = 6000; # Max gas production

# Read in bounds on injection
injectionbounds_df = CSV.read(joinpath(DATA_DIR,"injectionbounds.csv"), DataFrames.DataFrame, delim = " ", header = ["wellno", "low", "high"])
lb_inj = Containers.DenseAxisArray(injectionbounds_df[:,2], Wells);
ub_inj = Containers.DenseAxisArray(injectionbounds_df[:,3], Wells);

# Read in water cut for each well
wcut_df = CSV.read(joinpath(DATA_DIR,"wcut.csv"), DataFrames.DataFrame, delim = " ", header = ["wellno", "wcut"]);
wcut = Containers.DenseAxisArray(wcut_df[:,2], Wells);

# Read in GOR for each well
gor_df = CSV.read(joinpath(DATA_DIR,"gor.csv"), DataFrames.DataFrame, delim = " ", header = ["wellno", "gor"]);
gor = Containers.DenseAxisArray(gor_df[:,2], Wells);

# Read in datapoints from file for each well. Note that number of datapoints for each well differ
datapoints_df = CSV.read(joinpath(DATA_DIR,"datapoints.csv"), DataFrames.DataFrame, delim = " ", header = ["wellno", "pointno", "qinj", "qoil"]);

# welldatas = Array{DataFrame}(undef, N);
num_points = Array{Int64}(undef, N);
for w in 1:N
    welldata = datapoints_df[in.(datapoints_df.wellno, w), :]
    # @show welldata
    # welldatas[w] = welldata;
    num_points[w] =  nrow(welldata);
end

endpointno = accumulate(+, num_points);
variable_ranges = [endpointno[w]-num_points[w]+1:endpointno[w] for w in 1:N];


### Create model object
model = Model(GLPK.Optimizer);

### Define variables

# lambdas for all wells
@variable(model, lambdas[i = 1:nrow(datapoints_df)] >= 0);

# binary z_i for all wells
# z_i = 1 -> x ∈ [x_{i}, x_{i+1}]
@variable(model, z[i = 1:nrow(datapoints_df)], Bin);

# binary y_w for each well
# y_w = 1 -> production
# y_w = 0 -> no production 
@variable(model, y[i = 1:N], Bin);

### Define objective
@objective(model, Max, sum([dot(lambdas[variable_ranges[w]], datapoints_df.qoil[variable_ranges[w]]) for w in Wells]));

### Define constraints
for w in Wells
    ## constraints for convex combination ##
    
    @constraint(model, sum(lambdas[variable_ranges[w]]) == 1); # sum of lambdas in range for one well must be 1

    # sum of z_i in range for one well must equal 1
    @constraint(model, sum(z[variable_ranges[w]]) == 1);
    
    # IMPORTANT:
    # Remember that both λ and z variable start with index 1

    # λ_1 <= z_1 for all wells
    first_index_current_well = collect(variable_ranges[w])[1];
    @constraint(model, lambdas[first_index_current_well] <= z[first_index_current_well]);

    # λ_p <= z_{p-1}
    current_well_indices = collect(variable_ranges[w]);
    @constraint(model, lambdas[current_well_indices[end]] <= z[current_well_indices[end-1]]);
    
    # λ_i <= z_{i-1} + z_i
    for i in current_well_indices[2:end-1]
        @constraint(model, lambdas[i] <= z[i-1] + z[i]);
    end

    ## constraints from original problem ##

    # oil output is less than max and 0 if well is shut off (y_w = 0)
    # \hat{q_w^o} <= q^{max, w}_o * y_w
    # (2f in assignment)
    Qoilmax = maximum(datapoints_df.qoil[variable_ranges[w]]);
    @constraint(model, dot(lambdas[variable_ranges[w]], datapoints_df.qoil[variable_ranges[w]]) <= Qoilmax*y[w]);
    
    # qinj within bounds if well is on, if not (y_w = 0) then qinj = 0
    # (2g in assignment)
    @constraint(model, injectionbounds_df.low[w]*y[w] <= dot(lambdas[variable_ranges[w]], datapoints_df.qinj[variable_ranges[w]]));
    @constraint(model, dot(lambdas[variable_ranges[w]], datapoints_df.qinj[variable_ranges[w]]) <= injectionbounds_df.high[w]*y[w]);


end

## other higher level constraints from the original problem ##

# sum of qinj for all wells less than Qinj_max
# (2b in assignment)
@constraint(model, sum([dot(lambdas[var_range], datapoints_df.qinj[var_range]) for var_range in variable_ranges]) <= Qinj_max);

# sum of q_w and qoil for all wells <= Qliq_max.
# has been simplified from formulas
# (2c in assignment)  
@constraint(model, sum([dot(lambdas[variable_ranges[w]], datapoints_df.qoil[variable_ranges[w]])*(1/(1-wcut_df.wcut[w])) for w in 1:N]) <= Qliq_max);

# sum of qinj and q_g for all wells <= Qgas_max. 
# (2d in assignment)
@constraint(model, sum([ dot(lambdas[variable_ranges[w]], datapoints_df.qinj[variable_ranges[w]]) + gor_df.gor[w]*dot(lambdas[variable_ranges[w]], datapoints_df.qoil[variable_ranges[w]]) for w in 1:N]) <= Qgas_max);


optimize!(model);

@show value.(lambdas);
#@show value.(z);
@show value.(y);
@show objective_value(model);

@show termination_status(model);