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

open("mrp.amr.rawsentences.txt", "w") do io
    for amr in amr_dataset.amrset
           write(io, string(amr.input,"\n"))
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
