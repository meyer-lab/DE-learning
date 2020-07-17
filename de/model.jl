using OrdinaryDiffEq
using LinearAlgebra
using DiffEqSensitivity
using Optim
using Zygote
using DelimitedFiles

" Load the experimental data matrix. "
function get_data(path_RNAseq)
    # Import RNAseq data as 83 x 84 matrix preprocessed using python
    exp = DelimitedFiles.readdlm(path_RNAseq, ',', Float64)
    return Matrix(exp)
end

" Reshape a vector of parameters into the variables we know. "
@views function reshapeParams(p)
    w = reshape(p[1:6889], (83, 83))
    alpha = p[6890:6972]
    eps = p[6973:7055]
    
    @assert length(eps) == 83
    @assert length(alpha) == 83
    
    return w, alpha, eps
end

" Melt the variables back into a parameter vector. "
function unshapeParams(w, alpha, eps)
    return vcat(vec(w), alpha, eps)
end

" The ODE equations we're using. "
function ODEeq(du, u, p, t)
    w, alpha, epss = reshapeParams(p)
    du .= epss .* (1 .+ tanh.(w * u)) .- alpha .* u
    nothing
end


" The Jacobian of the ODE equations. "
function ODEjac(J, u, p, t)
    w, alpha, epss = reshapeParams(p)

    J .= diagm(epss .* (1 .- (tanh.(w * u) .^ 2))) * w - diagm(alpha)
    nothing
end


" Solve the ODE system. "
function solveODE(ps, tps=nothing)
    w, alpha, eps = reshapeParams(ps)
    u0 = eps ./ alpha #initial value

    if isnothing(tps)
        tspan = (0.0, 10000.0)
    else
        tspan = (0.0, maximum(tps))
    end

    fun = ODEFunction(ODEeq; jac=ODEjac)
    prob = ODEProblem(fun, u0, tspan, ps)
    sol = solve(prob, Tsit5(); sensealg = QuadratureAdjoint(; compile=true))

    if isnothing(tps)
        return last(sol)
    end

    return sol(tps)
end

" Remove the effect of one gene across all others to simulate the KO experiments. Returns parameters to be used in solveODE(). "
function simKO(pIn, geneNum)
    pIn = copy(pIn) # Need to copy as we're using views
    w, alpha, eps = reshapeParams(pIn)
    
    # Think we remove a column since this is the effect of one gene across all genes
    w[:, geneNum] .= 0.0
    
    pIn = unshapeParams(w, alpha, eps)
    
    return pIn
end

" Solves ODE system with given parameters to create comparable 83 x 84 matrix to experimental data. "
function sol_matrix(pIn)
    sol = ones(83, 84)
    for i = 1:83
        sol[:, i] = solveODE(simKO(pIn, i))
    end
    sol[:, 84] = solveODE(pIn)
    return sol
end

" Single simulation cost function. "
function sim_cost(pIn, exp_data)
    return norm(solveODE(pIn) .- exp_data)
end

" Returns SSE between model and experimental RNAseq data. "
function cost(pIn, exp_data)
    w = reshapeParams(pIn)[1]

    return norm(sol_matrix(pIn) - exp_data) + 0.01 * sum(abs.(w))
end

" Cost function gradient. Returns SSE between model and experimental RNAseq data. "
function costG!(G, pIn, exp_data)
    # negative control
    G .= Zygote.gradient(x -> sim_cost(x, exp_data[:, 84]), pIn)[1]

    for i = 1:83 # knockout simulations
        println(i)
        p_temp = simKO(pIn, i)
        g_temp = Zygote.gradient(x -> sim_cost(x, exp_data[:, i]), p_temp)[1]

        # Zero out corresponding parameters in gradient
        g_temp = simKO(g_temp, i)
        G .+= g_temp
    end

    # Regularization
    @. G[1:6889] += 0.01 * sign(pIn[1:6889])

    nothing
end


#optimize(ps -> cost(ps, e), costG!, ps, LBFGS(), Optim.Options(iterations = 10, show_trace = true))
