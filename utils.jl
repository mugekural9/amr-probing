function getent(x)
    for e in ents
       if e[2]==x; return e; end
    end
end

function getrel(x)
    for r in rels
       if r[2]==x; return r; end
    end
end

open("mrp2019training.amr.wsj.raw.txt", "w") do io
    for amr in amrset
      sent = udpipes[amr.id]
      write(io, string(sent,"\n"))
    end
end

function avgsentlength(l)
    sl = 0
    for amr in amr_dataset.amrset[1:l]
       sl+= length(split(amr.input," "))
    end
    return sl/l
end

avgsentlength(1000)



function find_amr(id) 
    for x in amr_dataset.amrset
        if x.id == id 
            return x
        end
    end
end

amr = find_amr("bolt12_07_4800.1")



function udpipe_reader()
  udpipes = Dict()
  file = "data/mrp/2019/companion/udpipe.mrp"
  amrset = []
  state = open(file);
  while true
      if eof(state)
          close(state)
          break
      else
          try
            labels = []
            udpipe_instance = JSON.parse(state)
            for n in udpipe_instance["nodes"]
              push!(labels, n["label"])
            end
            udpipes[udpipe_instance["id"]] = join(labels," ")
          catch
            @warn "Could not parse state. $state"
          end
      end
  end
  return udpipes
end

udpipes = udpipe_reader()