using OrdinaryDiffEq
using LinearAlgebra
using DiffEqSensitivity
using Optim
using Zygote

" Reshape a vector of parameters into the variables we know. "
@views function reshapeParams(p)
    w = reshape(p[1:9216], (96, 96))
    alpha = p[9217:9312]
    eps = p[9313:9408]
    
    @assert length(eps) == 96
    @assert length(alpha) == 96
    
    return w, alpha, eps
end

" Melt the variables back into a parameter vector. "
function unshapeParams(w, alpha, eps)
    return vcat(vec(w), alpha, eps)
end

" The ODE equations we're using. "
function ODEeq(du, u, p, t)
    w, alpha, eps = reshapeParams(p)
    
    du .= eps .* (1 .+ tanh.(w * u)) .- alpha .* u
end

" Solve the ODE system. "
function solveODE(ps)
    u0 = zeros(96)
    tspan = (0.0, 10000.0)
    prob = ODEProblem(ODEeq, u0, tspan, ps)
    sol = last(solve(prob, AutoTsit5(TRBDF2()); saveat = tspan[2], reltol=1e-8, abstol=1e-8))
    return sol
end
