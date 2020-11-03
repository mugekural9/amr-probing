using Knet
import Distributions: Uniform

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})

struct Probe
    w
end

function Probe(probedim::Int, embeddim::Int)
    w = param(rand(Uniform(-0.05,0.05), (probedim, embeddim)), atype=_atype)
    Probe(w)
end

function probe_distance(p, x, y)
    _, lossm = pred_distance(p, x, y)
    return lossm
end


function pred_distance(p, x, y)
    transformed = mmul(p.w, convert(_atype, x')) # P x T
    maxlength = size(transformed, 2)
    B = 1
    sentlengths = [maxlength]
    transformed = reshape(transformed, (size(transformed,1), maxlength, 1, B)) # P x T x 1 x B
    dummy = convert(_atype, zeros(1,1,maxlength,1))
    transformed = transformed .+ dummy   # P x T x T x B
    transposed = permutedims(transformed, (1,3,2,4))
    diffs = transformed - transposed
    squareddists = abs2.(diffs)
    squareddists = sum(squareddists, dims=1)  # 1 x T x T x B
    squareddists = reshape(squareddists, (maxlength, maxlength,B)) #  T x T x B

    y = reshape(y, (size(y,1), size(y,2),1)) # T x T x 1
    a = abs.(squareddists - convert(_atype, y))
    b = reshape(a, (size(a,1)*size(a,2),B))
    b = sum(b, dims=1)
    normalized_sent_losses = vec(b)./ convert(_atype, abs2.(sentlengths))
    batchloss = sum(normalized_sent_losses) /  B
    return squareddists, batchloss
end

function mmul(w, x)
    if w == 1 
        return x
    elseif w == 0 
        return 0 
    else 
        return reshape( w * reshape(x, size(x,1), :), 
                        (:, size(x)[2:end]...))
    end
end
