using OrdinaryDiffEq
using LinearAlgebra
using DiffEqSensitivity
using Optim
using Zygote
using ProgressMeter
using Zygote: @adjoint
import DelimitedFiles: readdlm, writedlm

" Load the experimental data matrix. "
function get_data(path_RNAseq)
    # Import RNAseq data as 83 x 84 matrix preprocessed using python
    return Matrix(readdlm(path_RNAseq, ',', Float64))
end

" Initialize the parameters based on the data. "
function initialize_params(exp)
    ɑ = fill(0.1, 83)
    ε = exp[:, 84] .* ɑ .+ 0.0001
    w = zeros(83, 83)

    return unshapeParams(w, ɑ, ε)
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

    du .= ε .* (1 .+ tanh.(w * u)) .- ɑ .* u
    nothing
end


" The Jacobian of the ODE equations. "
function ODEjac!(J, u, p, t)
    w, ɑ, ε = reshapeParams(p)

    J .= Diagonal(ε .* (sech.(w * u) .^ 2)) * w
    J[diagind(J)] .-= ɑ
    nothing
end

function ODEjac(u, p, t)
    w, ɑ, ε = reshapeParams(p)

    return Diagonal(ε .* (sech.(w * u) .^ 2)) * w .- Diagonal(ɑ)
end

" Regularization: Calculate cost as sum of all but largest complex components of eigenvalues. "
function costEigvals(pIn)
    @assert typeof(pIn) == Array{Float64,1}
    u = solveODE(pIn)
    @assert typeof(u) == Array{Float64,1}
    jacobian = ODEjac(u, pIn, 10000)
    im_comps = abs.(imag(eigen(jacobian).values))
    @assert typeof(im_comps) == Array{Float64,1}
    return norm(sum(im_comps) - maximum(im_comps))
end

@adjoint function LinearAlgebra.eigen(A::AbstractMatrix)
eV = eigen(A)
e,V = eV
n = size(A,1)
    eV, function (Δ)
        Δe, ΔV = Δ
        if ΔV === nothing
            (real.(inv(V)'*Diagonal(Δe)*V'), )
        elseif Δe === nothing
            F = [i==j ? 0 : inv(e[j] - e[i]) for i=1:n, j=1:n]
            (real.(inv(V)'*(F .* (V'ΔV))*V'), )
        else
            F = [i==j ? 0 : inv(e[j] - e[i]) for i=1:n, j=1:n]
            (real.(inv(V)'*(Diagonal(Δe) + F .* (V'ΔV))*V'), )
        end
    end
end

" Solve the ODE system. "
function solveODE(ps::AbstractVector{<:Number}, tps=nothing)
    @assert typeof(ps) == Array{Float64,1}
    w, ɑ, ε = reshapeParams(ps)
    u0 = ε ./ ɑ # initial value

    if isnothing(tps)
        tspan = (0.0, 10000.0)
    else
        tspan = (0.0, maximum(tps))
    end

    ODEfun = ODEFunction(ODEeq; jac=ODEjac!)
    senseALG = QuadratureAdjoint(; compile=true, autojacvec=ReverseDiffVJP(true))

    prob = ODEProblem(ODEfun, u0, tspan, ps)
    sol = solve(prob, AutoTsit5(TRBDF2()); sensealg=senseALG, reltol=1e-6)
    @assert typeof(last(sol)) == Array{Float64,1}
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
    sol = sol_matrix(pIn)
    costt = norm(sol - exp_data) + 1000 * (0.01 * norm(w, 1)) + 10000 * norm(w' * w - I) + costEigvals(pIn)
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
    @. G[1:6889] += 1000 * (0.01 * sign(pIn[1:6889]))
    w = reshapeParams(pIn)[1]
    T₀ = w' * w - I
    temp = 10000 * vec(2 / norm(T₀) * w * T₀)
    @. G[1:6889] += temp
    G += Zygote.gradient(costEigvals, pIn)[1]
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
