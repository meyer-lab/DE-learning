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
    exp = Matrix(exp)
    return exp
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
    w, alpha, eps = reshapeParams(p)
    
    du .= eps .* (1 .+ tanh.(w * u)) .- alpha .* u
end

" Solve the ODE system. "
function solveODE(ps)
    u0 = zeros(83)
    tspan = (0.0, 10000.0)
    prob = ODEProblem(ODEeq, u0, tspan, ps)
    sol = last(solve(prob, AutoTsit5(TRBDF2()); saveat = tspan[2], reltol=1e-8, abstol=1e-8))
    return sol
end

" Remove the effect of one gene across all others to simulate the KO experiments. "
function simKO(pIn, geneNum)
    pIn = copy(pIn) # Need to copy as we're using views
    w_temp, alpha, eps = reshapeParams(pIn)
    
    # Think we remove a column since this is the effect of one gene across all genes
    if geneNum == 1
        w = hcat(zeros(Float64, 83, 1), w_temp[:, 2:83])
    elseif geneNum == 83
        w = hcat(w_temp[:, 1:82], zeros(Float64, 83, 1))
        else geneNum != 1 || geneNum != 83
        w= hcat(w_temp[:, 1:geneNum - 1], zeros(Float64, 83, 1), w_temp[:, geneNum + 1:83])
    end
    
    pIn = unshapeParams(w, alpha, eps)
    
    return solveODE(pIn)
end

" Solves ODE system with given parameters to create comparable 83 x 84 matrix to experimental data. "
function sol_matrix(pIn)
    sol = ones(83, 84)
    for i = 1:83
        sol[:, i] = simKO(pIn, i)
    end
    sol[:, 84] = solveODE(pIn)
    return sol
end

" Cost function. Returns SSE between model and experimental RNAseq data. "
function cost(pIn)
    exp_data = get_data("./de/data/exp_data.csv")
    sse = 0
    for i = 1:83
        sol_temp = simKO(pIn, i)
        sse += sum((sol_temp .- exp_data[:, i]) .^ 2)
    end
    neg = solveODE(pIn)
    sse += sum((neg .- exp_data[:, 84]) .^ 2)
    return sse
end

" Calculates gradient of cost function. "
function g!(G, x)
    grads = Zygote.gradient(cost, x)
    G[:] .= grads[1]
end

" Single calculation of cost gradient. "
#ps = ones(7055)
#grads = Zygote.gradient(cost, ps)

" Run optimization. "
#optimize(cost, g!, ps, LBFGS(), Optim.Options(iterations = 10, show_trace = true))