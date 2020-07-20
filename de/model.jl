using OrdinaryDiffEq
using LinearAlgebra
using DiffEqSensitivity
using Optim
using Zygote
using ProgressMeter
import DelimitedFiles: readdlm

" Load the experimental data matrix. "
function get_data(path_RNAseq)
    # Import RNAseq data as 83 x 84 matrix preprocessed using python
    return Matrix(readdlm(path_RNAseq, ',', Float64))
end

" Initialize the parameters based on the data. "
function initialize_params(exp)
    alpha = fill(0.1, 83)
    epss = exp[:, 84] .* alpha .+ 0.0001
    w = zeros(83, 83)

    return unshapeParams(w, alpha, epss)
end

" Reshape a vector of parameters into the variables we know. "
@views function reshapeParams(p)
    w = reshape(p[1:6889], (83, 83))
    ɑ = p[6890:6972]
    ε = p[6973:7055]

    @assert length(ε) == 83
    @assert length(ɑ) == 83

    return w, ɑ, ε
end

" Melt the variables back into a parameter vector. "
function unshapeParams(w::AbstractMatrix, ɑ::AbstractVector, ε::AbstractVector)::Vector
    return vcat(vec(w), ɑ, ε)
end


" The ODE equations we're using. "
function ODEeq(du, u, p, t)
    w, ɑ, ε = reshapeParams(p)

    temp = w * u
    temp = map(tanh, temp)
    @. du = ε * (1 + temp) - ɑ * u
    nothing
end


" The Jacobian of the ODE equations. "
function ODEjac(J, u, p, t)
    w, ɑ, ε = reshapeParams(p)

    J .= Diagonal(ε .* (sech.(w * u) .^ 2)) * w
    J[diagind(J)] .-= ɑ
    nothing
end


" The Jacobian w.r.t. parameters. "
function paramjac(J, u, p, t)
    w, ɑ, ε = reshapeParams(p)

    # w.r.t. ɑ
    Ja = @view J[:, 6890:6972]
    Ja[diagind(Ja)] .= -u

    # w.r.t. ε
    Je = @view J[:, 6973:7055]
    Je[diagind(Je)] = 1 .+ tanh.(w * u)

    # w.r.t. w
    Jw = @view J[:, 1:6889]
    Jw = u' .* Diagonal(ε .* (sech.(w * u) .^ 2))

    nothing
end


const ODEfun = ODEFunction(ODEeq; jac=ODEjac, paramjac=paramjac)
const ODEalg = AutoDP5(TRBDF2(); stifftol=2.0, nonstifftol=2.0)
const senseALG = QuadratureAdjoint(; autojacvec=ReverseDiffVJP(true))


" Solve the ODE system. "
function solveODE(ps::AbstractVector{<:Number}, tps=nothing)
    w, ɑ, ε = reshapeParams(ps)
    u0 = ε ./ ɑ # initial value

    if isnothing(tps)
        tspan = (0.0, 10000.0)
    else
        tspan = (0.0, maximum(tps))
    end

    prob = ODEProblem(ODEfun, u0, tspan, ps)
    sol = solve(prob, ODEalg; sensealg=senseALG)

    if isnothing(tps)
        return last(sol)
    end

    return sol(tps)
end

" Remove the effect of one gene across all others to simulate the KO experiments. Returns parameters to be used in solveODE(). "
function simKO(pIn, geneNum)
    pIn = copy(pIn) # Need to copy as we're using views
    w, ɑ, ε = reshapeParams(pIn)
    
    # Think we remove a column since this is the effect of one gene across all genes
    w[:, geneNum] .= 0.0
    
    pIn = unshapeParams(w, ɑ, ε)
    
    return pIn
end

" Solves ODE system with given parameters to create comparable 83 x 84 matrix to experimental data. "
function sol_matrix(pIn::AbstractVector{<:Number})
    sol = ones(83, 84)
    for i = 1:83
        sol[:, i] = solveODE(simKO(pIn, i))
    end
    sol[:, 84] = solveODE(pIn)
    return sol
end

" Returns SSE between model and experimental RNAseq data. "
function cost(pIn, exp_data)
    w = reshapeParams(pIn)[1]
    costt = norm(sol_matrix(pIn) - exp_data) + 0.01 * norm(w, 1)
    println(costt)
    return costt
end

" Cost function gradient. Returns SSE between model and experimental RNAseq data. "
function costG!(G, pIn, exp_data)
    # negative control
    G .= Zygote.gradient(x -> norm(solveODE(x) - exp_data[:, 84]), pIn)[1]

    @showprogress 1 "Computing gradient..." for i = 1:83 # knockout simulations
        p_temp = simKO(pIn, i)
        g_temp = Zygote.gradient(x -> norm(solveODE(x) - exp_data[:, i]), p_temp)[1]

        # Zero out corresponding parameters in gradient
        G .+= simKO(g_temp, i)
    end

    # Regularization
    @. G[1:6889] += 0.01 * sign(pIn[1:6889])

    nothing
end

" Run the optimization. "
function runOptim(exp_data)
    x₀ = initialize_params(exp_data)
    func = ps -> cost(ps, exp_data)
    Gfunc = (a, b) -> costG!(a, b, exp_data)
    options = Optim.Options(iterations = 10, show_trace = true)
    x₋ = zeros(length(x₀))
    x₋[1:6889] .= -10.0
    x₊ = fill(1000.0, length(x₀))

    optt = optimize(func, Gfunc, x₋, x₊, x₀, Fminbox(LBFGS(), mu0=0.1), options)
    return optt
end
