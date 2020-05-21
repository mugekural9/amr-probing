include("probe.jl")
include("data.jl")

probe = Probe(1024,1024)
elmo_layer = 1
amr_file = "data/mrp/2019/training/amr/wsj.mrp"
amrset = AMRReader(amr_file, elmo_layer).amrset
