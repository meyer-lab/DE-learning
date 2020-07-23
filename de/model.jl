using OrdinaryDiffEq
using LinearAlgebra
using DiffEqSensitivity
using Optim
using Zygote
using ProgressMeter
import StatsFuns: softplus
import DelimitedFiles: readdlm, writedlm
using ReverseDiff: JacobianTape, jacobian!, compile

" Load the experimental data matrix. "
function get_data(path_RNAseq)
    # Import RNAseq data as 83 x 84 matrix preprocessed using python
    return Matrix(readdlm(path_RNAseq, ',', Float64))
end

" Initialize the parameters based on the data. "
function initialize_params(exp)
    ɑ = fill(0.1, 83)
    ε = exp[:, 84] .* ɑ .+ 0.0001

    return vcat(zeros(83*83), ɑ, ε)
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


" The ODE equations we're using. "
function ODEeq!(du, u, p, t)
    w, ɑ, ε = reshapeParams(p)
    du .= ε .* (1 .+ softplus.(w * u)) .- ɑ .* u
    nothing
end

const ODEeq_tape = compile(JacobianTape(ODEeq!, ones(83), (ones(83), ones(7055), 1.0)))


" Solve the ODE system. "
function solveODE(ps::AbstractVector{<:Number}, tps=nothing)
    w, ɑ, ε = reshapeParams(ps)
    u0 = ε ./ ɑ # initial value

    if isnothing(tps)
        tspan = (0.0, 10000.0)
    else
        tspan = (0.0, maximum(tps))
    end

    ODEfun = ODEFunction(ODEeq!; jac=(J, u, p, t) -> jacobian!(J, ODEeq_tape, (u, p, t)))
    senseALG = QuadratureAdjoint(; compile=true, autojacvec=ReverseDiffVJP(true))

    prob = ODEProblem(ODEfun, u0, tspan, ps)
    sol = solve(prob, AutoTsit5(TRBDF2()); sensealg=senseALG, reltol=1e-6)

    if isnothing(tps)
        return last(sol)
    end

    return sol(tps)
end


" Remove the effect of one gene across all others to simulate the KO experiments. Returns parameters to be used in solveODE(). "
function simKO!(pIn, geneNum)
    w, ɑ, ε = reshapeParams(pIn)

    # Remove a column, effect of one gene across all genes
    w[:, geneNum] .= 0.0
    return pIn
end

" Solves ODE system with given parameters to create comparable 83 x 84 matrix to experimental data. "
function sol_matrix(pIn::AbstractVector{<:Number})
    sol = ones(83, 84)
    for i = 1:83
        sol[:, i] = solveODE(simKO!(copy(pIn), i))
    end
    sol[:, 84] = solveODE(pIn)
    return sol
end

" Returns SSE between model and experimental RNAseq data. "
function cost(pIn, exp_data)
    w = reshapeParams(pIn)[1]
    costt = norm(sol_matrix(pIn) - exp_data) + 10 * (0.01 * norm(w, 1) + norm(w' * w - I)) # 10-fold stronger regularization
    println(costt)
    return costt
end

" Cost function gradient. Returns SSE between model and experimental RNAseq data. "
function costG!(G, pIn, exp_data)
    # negative control
    G .= Zygote.gradient(x -> norm(solveODE(x) - exp_data[:, 84]), pIn)[1]

    @showprogress 1 "Computing gradient..." for i = 1:83 # knockout simulations
        p_temp = simKO!(copy(pIn), i)
        g_temp = Zygote.gradient(x -> norm(solveODE(x) - exp_data[:, i]), p_temp)[1]

        # Zero out corresponding parameters in gradient
        G .+= simKO(g_temp, i)
    end

    # Regularization
    @. G[1:6889] += 10 * (0.01 * sign(pIn[1:6889]))
    w = reshapeParams(pIn)[1]
    T₀ = w' * w - I
    temp = 10 * vec(2 / norm(T₀) * w * T₀)
    @. G[1:6889] += temp

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
    x₊ = fill(100.0, length(x₀))

    optt = optimize(func, Gfunc, x₋, x₊, x₀, Fminbox(LBFGS()), options)
    return optt
end

" Save the optimized parameters "
function save_params(optt)
    println(Optim.minimum(optt))
    x = Optim.minimizer(optt)
    w, ɑ, ε = reshapeParams(x)
    writedlm( "./data/w.csv",  w, ',')
    writedlm( "./data/alpha.csv",  ɑ, ',')
    writedlm( "./data/epss.csv",  ε, ',')
end
