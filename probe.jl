_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})

struct Probe
    w
end


function Probe(probedim::Int, embeddim::Int)
    w = param(rand(Uniform(-0.05,0.05), (probedim, embeddim)), atype=_atype)
    Probe(w)
end

function (p::Probe)(x, y)
    transformed = p.w * convert(_atype, x)  # P x 1
    diffs = y - transformed 
    return loss = sum(abs2.(diffs))
end