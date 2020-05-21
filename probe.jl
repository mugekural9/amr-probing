using Knet, Distributions

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})

struct Probe
    w
end

function Probe(probedim::Int, embeddim::Int)
    w = param(rand(Uniform(-0.05,0.05), (probedim, embeddim)), atype=_atype)
    Probe(w)
end

function (p::Probe)(x, y, layer)
    transformed = mmul(p.w, convert(_atype, x))[:,:,layer+1]
    diffs = y - transformed 
    return loss = sum(abs2.(diffs))
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

