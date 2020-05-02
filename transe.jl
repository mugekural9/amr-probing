using Knet, LinearAlgebra
import Distributions: Uniform 


_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


struct GraphElements
    e
    l
end

function GraphElements(embed_dim::Int, num_entity::Int, num_rel::Int)
    k = 6/sqrt(embed_dim)
    e = param(rand(Uniform(-k, k), (embed_dim, num_entity)), atype=_atype)
    l = param(rand(Uniform(-k, k), (embed_dim, num_rel)), atype=_atype)
    GraphElements(e,l)
end


mutable struct triplet
    h
    l
    t 
end


function loss_triplet(h,l,t)
    return g.e[:,h]+g.l[:,l]-g.e[:,t]
end  


function generate_negatives(sbatch)
    trp_pairs = []
    for i in 1:length(sbatch)
        trp = sbatch[i]
        k = 0
        if i == 1; k = trp.t + 1; else; k = trp.t - 1; end
        corrupted_trp = triplet(trp.h, trp.l, k)
        push!(trp_pairs, (trp, corrupted_trp))
    end
    return trp_pairs
end

function loss_transe(crpairs, gamma)
    loss = 0
    for (trp, corrupted_trp) in crpairs
        pos_dis = loss_triplet(trp.h, trp.l, trp.t)
        neg_dis = loss_triplet(corrupted_trp.h, corrupted_trp.l, corrupted_trp.t)
        distance_diff = gamma + norm(pos_dis) - norm(neg_dis)
        loss += distance_diff
    end
    return loss
end


function pred_tail(h,l)
    h_and_r =g.e[:,h] + g.l[:,l]
    trans = h_and_r .- g.e
    norms = []
    for i in 1:size(trans,2)
       push!(norms,norm(trans[:,i]))
    end
    return argmin(norms)
end


function accuracy_tail(triplets)
    total_correct =  0
    for trp in triplets
        if trp.t == pred_tail(trp.h, trp.l)
            total_correct +=1
        end
    end
    return total_correct/length(triplets)
end

g = GraphElements(1024,4,2)
trp1 = triplet(1, 1, 2)
trp2 = triplet(2, 2, 3)
trp3 = triplet(3, 1, 4)
triplets = []
push!(triplets, trp1)
push!(triplets, trp2)
push!(triplets, trp3)
sbatch  =  triplets[1:3]
crpairs = generate_negatives(sbatch)
gamma = 0.2


function train()
    local i=0
    while accuracy_tail(triplets) != 1 # overfit to the graph
        i +=1
        J = @diff loss_transe(crpairs, gamma)
        println("iteration: $i, loss: $J")
        for par in params(g)
            g = grad(J, par)
            update!(value(par), g, eval(Meta.parse("Adam()")))
        end
    end
end

train()

