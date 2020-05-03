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
