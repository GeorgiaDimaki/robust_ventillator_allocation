# robust_ventillator_allocation
Data driven robust allocation model that adjusts the demand uncertainty using daily available demand data.

## Robust Ventilator State Allocation

The ventilator state allocation in US is defined as the decision problem of ventilator transfers across US states in order to optimally share the scarce resource of ventilators during a pandemic (COVID19 in this case study) in order to decrease deaths related to ventilator shortage accross the nation.

Basic challenge of this sharing is the uncertainty related with ventilator demand day by day. Unfortunatelly, in case of pandemics of not studied deseases, it is hard to estimate the impact on resources, as for example the ventilators, since there are not enough recorded data of even similarly spreading deseases.

Apart from the inherent uncertainty, we should also notice that decision making should happen in advance in order for the transfers to happen on time. Not only that but also the allocation should be continuous in order to have an impact. To address these challenges we propose a data driven adptive robust allocation model that adjusts the demand uncertainty using daily available demand data. Specifically, the model keeps improving as long as historical data accumulate. In this way we do not need to make assumptions on the demand, but we learn it instead. This results into more robust and stable solutions. Finally the model is adaptive so that it is able to look "days ahead" before deciding on transfers but without being impacted by future uncertainty.

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
          
where U is the uncertainty set discribed in the project_report.pdf

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

_Note: The formulation is based on the
https://github.com/COVIDAnalytics/ventilator-allocation
with the following simplifications:
  - No shortfall buffer is used due to the use of uncertainty sets_

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
