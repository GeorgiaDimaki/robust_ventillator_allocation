#-----------------------------------------------------------------------
# Resources:
# [1] http://github.com/IainNZ/JuMPeR.jl
#       -- example/inventory.jl
# [2] A. Ben-Tal, A. Goryashko, E. Guslitzer, A. Nemirovski
#    "Adjustable Robust Solutions of Uncertain Linear Programs"
# [3] https://github.com/COVIDAnalytics/ventilator-allocation
#-----------------------------------------------------------------------

"""
Robust Ventilator State Allocation

The ventilator state allocation in US is defined as the decision problem
of ventilator transfers across US states in order to optimally share the
scarce resource of ventilators during a pandemic (COVID19 in this case study)
in order to decrease deaths related to ventilator shortage accross the nation.

Basic challenge of this sharing is the uncertainty related with ventilator
demand day by day. Unfortunatelly, in case of pandemics of not studied deseases,
it is hard to estimate the impact on resources, as for example the ventilators,
since there are not enough recorded data of even similarly spreading deseases.

Apart from the inherent uncertainty, we should also notice that decision making
should happen in advance in order for the transfers to happen on time. Not only
that but also the allocation should be continuous in order to have an impact.

To address these challenges we propose a data driven adptive robust
allocation model that adjusts the demand uncertainty using daily available
demand data. Specifically, the model keeps improving as long as historical
data accumulate. In this way we do not need to make assumptions on the demand,
but we learn it instead. This results into more robust and stable solutions.
Finally the model is adaptive so that it is able to look "days ahead" before
deciding on transfers but without being impacted by future uncertainty.

The particular problem solved is

    min ∑ᵢₜ VSᵢₜ + λ (∑ᵢₜ (fᵢₜ + ∑ⱼ xᵢⱼₜ))

    s.t.  ∀i ∈ S, ∀t ∈ T:

          xᵢᵢₜ = 0
          Vᵢ₀  = vᵢ
          Vᵢₜ  ≥ β*vᵢ
    VSᵢₜ + Vᵢₜ ≥ dᵢₜ   ∀ d ∈ U # Worst-case demand
          Vᵢₜ  = Vᵢ₍ₜ₋₁₎ + fᵢ₍ₜ₋ₖ₎ + ∑ᵢ xᵢⱼ₍ₜ₋₁₎ - ∑ⱼ xᵢⱼₜ
          Vᵢₜ  ≥ ∑ₕ (fᵢₕ + ∑ᵢ xᵢⱼₕ) , max(1, t-min_days)≤ h ≤ t-1
        ∑ᵢ fᵢₜ ≤ F
          VSᵢₜ ≤  Dmax * yᵢₜ
       ∑ⱼ xᵢⱼₜ ≤  Vᵢ₍ₜ₋₁₎*(1-yᵢₜ)

          VS, x, f   ≥ 0
          yᵢₜ ∈ {0,1}

where U is the uncertainty set discribed in the report

Disclaimers:

Actual use of this model would require a centralized management of data and
policy making that is not yet in implemented. For this reason the following
assumptions are used:

    - A1: There is a factory which makes per day F ventilators and acts as
          a federal supply of new ventilators to states
    - A2: The states can decide to contribute in the sharing system an amount
          of ventilators (1-β)*100% from what they initialy have
    - A3: The maximum number of days to transfer a ventilator is 3. Typically
          this depends on the distance between states, but for simplicity
          it is assumed to be the same for all state distances.

Note: The formulation is based on the
https://github.com/COVIDAnalytics/ventilator-allocation
with the following simplifications:

- No shortfall buffer is used due to the use of uncertainty sets

Baseline: All reality known in advance
Solve the proposed model with the buffer.

Sliding horizon:

Start on day 2 using data from day 1.

current day = 2
1) Form uncertainty sets from data from day 1 to day current_day - 1
2) Solve static RO for 4 days in advance
3) Implement day <current_day>: Get new Ventilator supplies
4) Realize the demand of current_day.
5) current_day += 1
6) Go to step 1

"""
#--- Libraries

using Distributions,LinearAlgebra  # general
using JuMP, Gurobi                                  # optimization
using CSV, DataFrames, DataFramesMeta               # data manipulation
using Dates

#--- Load data
#
# The data and parameters used follow the notebook provided by
# the COVIDAnalytics team here:
# https://github.com/COVIDAnalytics/ventilator-allocation/blob/master/notebooks/Allocation_byState.ipynb
#
# For the purpose of this project I cloned the repository and I use directly the
# data they use for the **ihme** predictions

include("ventilator-allocation-master/src/utils_load.jl")

model_choice = "ihme" # altervatively ode

# load the states
states = load_states(model_choice);

# load the projected demand
demands, demand_days = demand_matrix(model_choice)
demands = demands[:,1:35] # demand data used

# load base supply
base_supply = round.(Int,load_supply(states))

# create distance matrix
distances_full = CSV.read("~/Documents/15.094/ventilator-allocation-master/processed/state_distances.csv")
distances = @where(distances_full, [x in states for x in :Column1])[Symbol.(states)]
distances = distances |> Matrix

#--- Parameters set up (Only relevant explained)
#
# Surge supply --> factory supply. As you can see the value is fixed to 450 units
#                  same as the model's assumptions
# delays --> time it takes to transfer across states. Fixed to 3 which is the
#            maximum time.
# The rest you can see in the function call and specifically get the values:
# base_supply = base_supply .* 0.5
# lasso = 1e-1
# minimum_stock_fraction = 0.8
# max_ship_per_day = 3000.0
# vent_days = 10

surge_supply = zeros(size(demands, 2));
surge_supply = 450.0 * ones(size(demands, 2))
surge_supply[1:3] .= 0.0
surge_supply[33:end] .= 0.0

delays = 3 * ones(Int, size(distances));

global GUROBIENV = Gurobi.Env()

#--- Actual Shortfall

function actual_shortfall(demands::Matrix, base_supply::Vector)
    D = size(demands, 2)
    S = size(base_supply, 1)
    # total number of ventilator shortfall for all the period
    total = sum([max(0, demands[s, d] - base_supply[s]*0.5) for d=1:D for s=1:S])
    # number of days the states had shortfall
    days = sum([sum([max(0, demands[s, d] - base_supply[s]*0.5) for s=1:S]) > 0 for d=1:D])
    return total, days
end

total, days = actual_shortfall(demands[:,3:35], base_supply)
total # 46503
days # 35


#--- Ventilator allocation as proposed (buffer version)

"""
    Main allocation function
    Args:
        - demands:       a (# states) × (# days) matrix; element (i, j) is the demand for ventilators in state i for period j
        - env:           Gurobi Environment to minimize the number of annoying license messages
        - base_supply:   the initial supply of ventilators in each state
        - surge_supply:  the total amount of ventilators that can be added into the system by FEMA on each day
        - distances:     the matrix of distances between all pairs of states
        - delays:        the matrix of integer delays required to ship between all pairs of states
    Keyword args:
        - lasso:         the LASSO penalty coefficient on transfers (exact sparse will not scale)
        - alpha:         robustness buffer above the demand
        - rho:           relative cost of missing buffer demand vs. real demand
        - minimum_stock_fraction: fraction of base ventilator stock that cannot leave a state
        - max_ship_per_day: maximum number of ventilators that can be sent out by a state per day
        - easy_incentive: whether to guarantee hat states will be no worse off by participating
        - gurobiargs:    Gurobi keyword arguments
"""
function allocate_ventilators(demands::Matrix, base_supply::Vector, surge_supply::Vector,
                              distances::Matrix, delays::Matrix{Int}, env=GUROBIENV;
                              lasso::Real=0.0, minimum_stock_fraction::Real = 0.0,
                              only_send_excess::Bool=false, easy_incentive::Bool=false, max_ship_per_day::Real = 3000.0,
                              vent_days::Int = 0, alpha::Real=0.1, rho::Real=0.25,
                              lp_relaxation::Bool=false,
                              gurobiargs...)
    # numbers of days
    D = size(demands, 2)
    # number of states
    S = size(demands, 1)
    model = Model(solver=GurobiSolver(env; gurobiargs...))
    # Supply of ventilators in each state, at each time, cannot go below a certain threshold
    @variable(model, supply[s=1:S, 0:D] >= floor.(minimum_stock_fraction * base_supply[s]))
    # amount of flow sent from state i to state j on day d
    if lp_relaxation
        @variable(model, flow[1:S, 1:S, 1:D] >= 0)
        @variable(model, surge[1:S, 1:D] >= 0)
    else
        @variable(model, flow[1:S, 1:S, 1:D] >= 0, Int)
        @variable(model, surge[1:S, 1:D] >= 0, Int)
    end
    # shortfall of demand in each state
    @variable(model, shortfall[1:S, 1:D] >= 0)
    # buffer for shortage in each state
    @variable(model, buffer[1:S, 1:D] >= 0)
    # additional ventilators provided to each state from national surge supply

    # initial supply
    @constraint(model, [s=1:S], supply[s, 0] == base_supply[s])
    # Surge supply constraints
    @constraint(model, [d=1:D], sum(surge[s, d] for s=1:S) <= surge_supply[d])
    # no self-flows
    @constraint(model, [s=1:S, d=1:D], flow[s, s, d] == 0)
    # flow constraints
    @constraint(model, [s=1:S, d=1:D],
                supply[s, d] == supply[s, d - 1]                            # amount yesterday
                                + surge[s, d] # surge supply
                                - sum(flow[s, dest, d] for dest = 1:S) # shipments leaving today
                                + sum(flow[orig, s, d - delays[orig, s]] for orig = 1:S if d > delays[orig, s])) # incoming
    # shortfall
    @constraint(model, [s=1:S, d=1:D], shortfall[s, d] + supply[s,d] + buffer[s,d] >= demands[s,d] * (1 + alpha))
    @constraint(model, [s=1:S, d=1:D], shortfall[s, d] + supply[s,d] >= demands[s, d])
    # max size of total outward transfers
    @constraint(model, [s=1:S, d=1:D], sum(flow[s, out, d] for out = 1:S) <= max_ship_per_day)

    # can't go below inflow amount within 10 days: if receive X on day d, must have at least X supply for day d+1,...d+10
    if vent_days != 0
        # @constraint(model, [s=1:S, d=1:D-vent_days, d_lag=d+1:d+vent_days], sum(flow[i, s, d] for i = 1:S) <= supply[s, d_lag])
        @constraint(model, [s=1:S, d=2:D],
                    sum(flow[i, s, d_in] for i = 1:S, d_in=max(d-vent_days,1):(d-1)) +
                    sum(surge[s, d_in] for d_in=max(d-vent_days,1):(d-1)) <= supply[s, d])
    end

    if easy_incentive
        max_shortfalls = shortfall_without_pooling(demands, base_supply, alpha, rho)
        @constraint(model, [s=1:S], sum(shortfall[s, d] for d=1:D) <= max_shortfalls[s])
    end

    if only_send_excess
        # binary variables that indicate whether there is a shortfall
        @variable(model, has_shortfall[1:S, 1:D], Bin)
        # big-M constraint
        @constraint(model, [s=1:S, d=1:D], shortfall[s, d] + buffer[s, d] <= demands[s, d] * (1 + alpha) * has_shortfall[s, d])
        # no flow if shortfall
        @constraint(model, [s=1:S, d=1:D], sum(flow[s, out, d] for out = 1:S) <= (1 - has_shortfall[s, d]) * max_ship_per_day)
    end

    # objective: minimize shortfall subject to LASSO penalty on flows
    @objective(model, Min, sum(shortfall[s, d] for s=1:S, d=1:D) +
                           rho * sum(buffer[s, d] for s=1:S, d=1:D) +
                           lasso * (sum((distances[s1, s2] + 10) * flow[s1, s2, d] for s1 = 1:S, s2 = 1:S, d = 1:D)
                                        + 10 * sum(surge[s, d] for s=1:S, d=1:D)))

    solve(model)
    supply_levels = [getvalue(supply[s, d]) for s=1:S, d=1:D]
    transfers = [getvalue(flow[s1, s2, d]) for s1=1:S, s2=1:S, d=1:D]
    surge = [getvalue(surge[s, d]) for s=1:S, d=1:D]

    return supply_levels, transfers, surge
end

#--- Results on proposed model

@time supply, transfers, surge = allocate_ventilators(demands[:,3:35], base_supply .* 0.5, surge_supply, distances,
    delays, lasso=1e-1, only_send_excess=true, minimum_stock_fraction = 0.8, easy_incentive = false,
    max_ship_per_day = 3000.0, vent_days = 10,
    OutputFlag=1, TimeLimit=120);

function calculate_results(demands, base_supply, transfers, surge, delays)
    D = size(demands, 2)
    S = size(base_supply, 1)
    supply = base_supply
    total = 0
    days = 0
    for d=1:D
        for v=1:S
            supply[v] += surge[v,d] - sum(transfers[v,:,d])
            if d > 3
                supply[v]+= sum(transfers[:,v,d- 3])
            end
        end
        diff = sum(max(0,demands[s,d] - supply[s]) for s=1:S)
        if diff > 0
            total += diff
            days += 1
        end
    end
    return total, days, sum(transfers), sum(surge), supply
end

total, days, total_tr, total_f, _ = calculate_results(demands[:,3:35], base_supply*0.5, transfers, surge, delays)
# total = 3356
# days = 4
# total transfers = 1190
# total transfers from federal = 1209
# total transfers = 2399


#--- Ventilator allocation Baseline model (without buffer - robustness)

function allocate_ventilators_baseline(demands::Matrix, base_supply::Vector, surge_supply::Vector,
                              distances::Matrix, delays::Matrix{Int}, env=GUROBIENV;
                              lasso::Real=0.0, minimum_stock_fraction::Real = 0.0,
                              max_ship_per_day::Real = 3000.0,
                              vent_days::Int = 0,
                              lp_relaxation::Bool=false,
                              gurobiargs...)
    # numbers of days
    D = size(demands, 2)
    # number of states
    S = size(demands, 1)
    model = Model(solver=GurobiSolver(env; gurobiargs...))
    # Supply of ventilators in each state, at each time, cannot go below a certain threshold
    @variable(model, supply[s=1:S, 0:D] >= floor.(minimum_stock_fraction * base_supply[s]))
    # amount of flow sent from state i to state j on day d
    if lp_relaxation
        @variable(model, flow[1:S, 1:S, 1:D] >= 0)
        @variable(model, surge[1:S, 1:D] >= 0)
    else
        @variable(model, flow[1:S, 1:S, 1:D] >= 0, Int)
        @variable(model, surge[1:S, 1:D] >= 0, Int)
    end
    # shortfall of demand in each state
    @variable(model, shortfall[1:S, 1:D] >= 0)
    # buffer for shortage in each state
    @variable(model, buffer[1:S, 1:D] >= 0)
    # additional ventilators provided to each state from national surge supply

    # initial supply
    @constraint(model, [s=1:S], supply[s, 0] == base_supply[s])
    # Surge supply constraints
    @constraint(model, [d=1:D], sum(surge[s, d] for s=1:S) <= surge_supply[d])
    # no self-flows
    @constraint(model, [s=1:S, d=1:D], flow[s, s, d] == 0)
    # flow constraints
    @constraint(model, [s=1:S, d=1:D],
                supply[s, d] == supply[s, d - 1]                            # amount yesterday
                                + surge[s, d] # surge supply
                                - sum(flow[s, dest, d] for dest = 1:S) # shipments leaving today
                                + sum(flow[orig, s, d - delays[orig, s]] for orig = 1:S if d > delays[orig, s])) # incoming
    # shortfall
    @constraint(model, [s=1:S, d=1:D], shortfall[s, d] + supply[s,d] >= demands[s, d])
    # max size of total outward transfers
    @constraint(model, [s=1:S, d=1:D], sum(flow[s, out, d] for out = 1:S) <= max_ship_per_day)

    # can't go below inflow amount within 10 days: if receive X on day d, must have at least X supply for day d+1,...d+10
    if vent_days != 0
        # @constraint(model, [s=1:S, d=1:D-vent_days, d_lag=d+1:d+vent_days], sum(flow[i, s, d] for i = 1:S) <= supply[s, d_lag])
        @constraint(model, [s=1:S, d=2:D],
                    sum(flow[i, s, d_in] for i = 1:S, d_in=max(d-vent_days,1):(d-1)) +
                    sum(surge[s, d_in] for d_in=max(d-vent_days,1):(d-1)) <= supply[s, d])
    end

    # binary variables that indicate whether there is a shortfall
    @variable(model, has_shortfall[1:S, 1:D], Bin)
    # big-M constraint
    @constraint(model, [s=1:S, d=1:D], shortfall[s, d] <= demands[s, d] * has_shortfall[s, d])
    # no flow if shortfall
    @constraint(model, [s=1:S, d=1:D], sum(flow[s, out, d] for out = 1:S) <= (1 - has_shortfall[s, d]) * max_ship_per_day)


    # objective: minimize shortfall subject to LASSO penalty on flows
    @objective(model, Min, sum(shortfall[s, d] for s=1:S, d=1:D) +
                           lasso * (sum((distances[s1, s2] + 10) * flow[s1, s2, d] for s1 = 1:S, s2 = 1:S, d = 1:D)
                                        + 10 * sum(surge[s, d] for s=1:S, d=1:D)))

    solve(model)
    supply_levels = [getvalue(supply[s, d]) for s=1:S, d=1:D]
    transfers = [getvalue(flow[s1, s2, d]) for s1=1:S, s2=1:S, d=1:D]
    surge = [getvalue(surge[s, d]) for s=1:S, d=1:D]

    return supply_levels, transfers, surge
end

#--- Results on baseline model

@time supply, transfers, surge = allocate_ventilators_baseline(demands[:,3:35], base_supply .* 0.5,
    surge_supply, distances,
    delays, lasso=1e-1, minimum_stock_fraction = 0.8,
    max_ship_per_day = 3000.0, vent_days = 10,
    OutputFlag=1, TimeLimit=120);


total, days, total_tr, total_f, _ = calculate_results(demands[:,3:35], base_supply*0.5, transfers, surge, delays)
# total = 6666
# days = 4
# total transfers = 1148
# total transfers from federal = 972
# total transfers = 1220

#--- RO model using the data driven uncertainty sets described

# Note: this is using
# JuMP 0.18.6
# JuMPeR 0.6.0
# Julia 1.4.0

using JuMPeR

function allocate_ventilators_ro(demands, base_supply, surge_supply, distances, delays,
                              lasso=0.0, horizon=4, current_day=4,
                              minimum_stock_fraction = 0.0,
                              max_ship_per_day = 3000.0,
                              vent_days = 0, Γ=1)

    ## Set parameters
    # UNCERTAINTY_SET = :CLT
    # setting horizon
    T = horizon

    # number of states
    S = size(demands, 1)
    # max new ventilators per day
    F = surge_supply

    model = RobustModel(solver=GurobiSolver( OutputFlag=1, TimeLimit=120))
    #---- Uncertainty Sets

    # The uncertain parameters are the ventilator demands


    # @uncertain(model, d[1:S, 1:T])
    # Γ1 = Γ2 = 3
    #
    # μ1 = vec(round.(Int,mean(demands[:,1:current_day - 1], dims=2)))  # Want a column vector
    # σ1 = vec(round.(Int,std(demands[:,1:current_day - 1], dims=2)))
    #
    # @constraint(model, [s=1:S], sum(d[s,:]) - T*μ1[s] <= Γ1*sqrt(T)*σ1[s])
    # @constraint(model, [s=1:S], sum(d[s,:]) - T*μ1[s] >= -Γ1*sqrt(T)*σ1[s])
    #
    # μ2 = round(Int,mean(sum(demands[s,1:current_day - 1] for s = 1:S)))
    # σ2 = round(Int,std(sum(demands[s,1:current_day - 1] for s = 1:S)))
    #
    # @constraint(model,  sum(d)/S - T*μ2 <= Γ2*sqrt(T)*σ2)
    # @constraint(model, sum(d)/S - T*μ2 >= -Γ2*sqrt(T)*σ2)

    @uncertain(model, d[1:S, 1:T])


    # if current_day > 1
    μ2 = round(Int,mean(sum(demands[s,1:current_day - 1] for s = 1:S)))  # Want a column vector
    σ2 = round(Int,std(sum(demands[s,1:current_day - 1] for s = 1:S)))


    # the total demand across the days should follow the mean demand in US
    @constraint(model, norm([sum(d[:, t]) for t=1:T] .- μ2,1) <= Γ*sqrt(T)*σ2)

    μ1 = vec(round.(Int,mean(demands[:,1:current_day - 1], dims=2)))  # Want a column vector
    σ1 = vec(round.(Int,std(demands[:,1:current_day - 1], dims=2)))

    # the demand across days for every state should be close to the mean for the state
    @constraint(model, [s=1:S], norm(d[s,:] .- μ1[s],1) <= Γ*sqrt(T)*σ1[s])

    # else
    #     # due to unknown future, take a conservative approach
    #     # expected to redistribute ventilators in US.
    #     # If bound too low ==> shortfalls
    #     # If bound too high ==> unwanted transfers
    #     bound = 500
    #     @constraint(model, [s=1:S, t=1:T], d[s, t] <= bound)
    # end

    #---- Variables

    # Supply of ventilators in each state, at each time, cannot go below a certain threshold
    @variable(model, V[s=1:S, 0:T] >= floor.(minimum_stock_fraction * base_supply[s]))
    # shortfall of demand in each state
    @variable(model, VS[1:S, 1:T] >= 0)
    # amount of flow sent from state i to state j on day d
    if true
        @variable(model, x[1:S, 1:S, 1:T] >= 0)
        @variable(model, f[1:S, 1:T] >= 0)
    else
        @variable(model, x[1:S, 1:S, 1:T] >= 0, Int)
        @variable(model, f[1:S, 1:T] >= 0, Int)
    end

    # binary variables that indicate whether there is a shortfall
    @variable(model, y[1:S, 1:T], Bin)

    #---- Constraints


    # Surge supply constraints
    @constraint(model, [t=1:T], sum(f[s, t] for s=1:S) <= F)

    # no self-flows
    @constraint(model, [s=1:S, t=1:T], x[s, s, t] == 0)

    # initialize supplies to their baseline levels
    @constraint(model, [s=1:S], V[s, 0] == base_supply[s])

    # flow constraints
    @constraint(model, [s=1:S, t=1:T],
            V[s, t] == V[s, t - 1]                            # amount yesterday
                    + f[s, t]                                 # surge supply
                    - sum(x[s, dest, t] for dest = 1:S) # shipments leaving today
                    + sum(x[orig, s, max(1,t - delays[orig, s])] for orig = 1:S if t > 1)) # incoming

    # shortfall definition constraints
    @constraint(model, [s=1:S, t=1:T], VS[s, t] + V[s,t] >= d[s, t])

    # can't go below inflow amount within 10 days: if receive X on day d, must have at least X supply for day d+1,...d+10
    if vent_days != 0
        @constraint(model, [s=1:S, t=1:T],
                    sum(distances[i,s]*x[i, s, d_in] for i = 1:S, d_in=max(t-vent_days,1):(t-1)) +
                    sum(f[s, d_in] for d_in=max(t-vent_days,1):(t-1)) <= V[s, t])
    end

    # big-M constraint
    @constraint(model, [s=1:S, t=1:T], VS[s, t] <= 100000000000000000 * y[s, t])
    # no flow if shortfall
    @constraint(model, [s=1:S, t=1:T], sum(x[s, out, t] for out = 1:S) <= (1 - y[s,t]) * max_ship_per_day)
    # instead of max_ship_per_day alternative : V[s,d-1]

    #---- Objective
    # objective: minimize shortfall subject to LASSO penalty on flows
    @objective(model, Min,  sum(VS[s, t] for s=1:S, t=1:T) +
                                lasso * (sum((distances[s1, s2] + 10) * x[s1, s2, t] for s1 = 1:S, s2 = 1:S, t = 1:T)
                                + 10 * sum(f[s, t] for s=1:S,t=1:T)))

    solve(model)

    supply_levels = [getvalue(V[s, 1]) for s=1:S]
    transfers = [getvalue(x[s1, s2, 1]) for s1=1:S, s2=1:S]
    surge = [getvalue(f[s,1]) for s=1:S]

    return supply_levels, transfers, surge
end


function sliding_window_simulate(demands, base_supply, distances, window, start_day, Γ)
    # setting parameters
    F = 450
    lasso = 0.1
    horizon = window
    max_fraction = 0.8
    max_shippment = 3000.0
    vent_days = 10
    delays = 3 * ones(Int, size(distances));
    days = size(demands, 2)
    S = size(demands, 1)

    total_sh = 0
    total_sh_days = 0
    total_transfers = zeros(S,S,start_day-1)
    total_f = zeros(S,start_day-1)
    vents = base_supply


    for day=start_day:days
        @time _, transfers, surge = allocate_ventilators_ro(demands,vents, F, distances, delays, lasso, horizon,day, max_fraction, max_shippment, vent_days, Γ);
        total_f = hcat(total_f, surge)
        total_transfers = cat(dims=3, total_transfers, transfers)

        for v=1:S
            vents[v] += total_f[v,day] - sum(total_transfers[v,:,day])
            if day > 3
                vents[v]+= sum(total_transfers[:,v,day- 3])
            end
        end
        diff = sum(max(0,demands[s,day] - vents[s]) for s=1:S)
        if diff > 0
            total_sh += diff
            total_sh_days += 1
        end
    end

    tr = sum(total_transfers)
    f = sum(total_f)

    return total_sh, total_sh_days, tr, f, vents
end

total_sh, total_sh_days, total_transfers, total_f, _ = sliding_window_simulate(demands, base_supply*0.5, distances, 4,3, 0)

# Γ = 0
# total = 21544
# days = 23
# total transfers = 639
# total transfers from federal = 1161
# total transfers = 1800

total_sh, total_sh_days, total_transfers, total_f, _ = sliding_window_simulate(demands, base_supply*0.5, distances, 4,3, 1/sqrt(33))

# Γ = 1/sqrt(T) -- T = 35-2 = 33
# total = 4608
# days = 21
# total transfers = 1616
# total transfers from federal = 1849
# total transfers = 3465

total_sh, total_sh_days, total_transfers, total_f, _ = sliding_window_simulate(demands, base_supply*0.5, distances, 4,3, 1)

# Γ = 1
# total = 3710
# days = 5
# total transfers = 3808
# total transfers from federal = 2632
# total transfers = 6430

total_sh, total_sh_days, total_transfers, total_f, _ = sliding_window_simulate(demands, base_supply*0.5, distances, 4,3, 2)

# Γ = 2
# total =  3670
# days = 5
# total transfers = 7160
# total transfers from federal = 3564
# total transfers = 10624

total_sh, total_sh_days, total_transfers, total_f, _ = sliding_window_simulate(demands, base_supply*0.5, distances, 4,3, 2.5)

# Γ = 2.5
# total =  3937
# days = 4
# total transfers = 8490
# total transfers from federal =  4888
# total transfers = 13378

total_sh, total_sh_days, total_transfers, total_f, _ = sliding_window_simulate(demands, base_supply*0.5, distances, 4,3, 3)

# Γ = 3
# total =  3940
# days = 5
# total transfers = 11063
# total transfers from federal = 6840
# total transfers = 11903
