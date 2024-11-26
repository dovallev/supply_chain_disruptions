import pyomo.environ as pyo
import itertools

def flatten_tuple(tup, flat=[]):
    for i in tup:
        if type(i) == tuple or type(i) == list:
            flatten_tuple(i, flat=flat)
        else:
            flat.append(i)        
    return tuple(flat)


def disruption(m, var, sets, bnd, dis_name):

    m.add_component(dis_name, pyo.ConstraintList())

    lists = ([s.data() for s in sets])
    combinations = itertools.product(*lists)

    for idx in combinations:
        idx = flatten_tuple(idx, flat=[])
        getattr(m, dis_name).add(expr=pyo.inequality(bnd[idx][0], var[idx], bnd[idx][1]))

    return m

# P_third = {(p,r,t): (0, m.P_bnd[p,r,t][1]/3) for (p,r) in m.PRtuples for t in m.s_T}
# m = disruption(m, var=m.P, sets=[m.PRtuples, m.s_T], bnd=P_third, dis_name='P_third')


def plant_disruption(m, plant, time, off, label, preparation=False):
    m.add_component(label, pyo.ConstraintList())
    
    if  m.time_production:
        Pvar = m.Pin
    else:
        Pvar = m.P

    for t in m.s_T:
        for (p, r) in m.PRtuples:
            if not preparation:
                if p == plant and t <= (1 - time) * m.s_T.at(-1):
                    getattr(m, label).add(expr=Pvar[p,r,t] <= m.P_bnd[p,r,t][1] * off)
            else:
                if p == plant and t >= (1 - time) * m.s_T.at(-1):
                    getattr(m, label).add(expr=Pvar[p,r,t] <= m.P_bnd[p,r,t][1] * off)
    return m

def arc_disruption(m, arc, time, off, label, preparation=False):
    m.add_component(label, pyo.ConstraintList())

    for t in m.s_T:
        for (mtl, nin, nout, tp) in m.MAtuples:
            if not preparation:
                if (nin, nout, tp) == arc and t <= (1 - time) * m.s_T.at(-1):
                    getattr(m, label).add(expr=m.Fin[mtl,nin,nout,tp,t] <= m.F_bnd[mtl,nin,nout,tp,t][1] * off)
            else:
                if (nin, nout, tp) == arc and t >= (1 - time) * m.s_T.at(-1):
                    getattr(m, label).add(expr=m.Fin[mtl,nin,nout,tp,t] <= m.F_bnd[mtl,nin,nout,tp,t][1] * off)
    return m

def inventory_disruption(m, place, time, off, label, preparation=False):
    m.add_component(label, pyo.ConstraintList())

    for t in m.s_T:
        for (mtl, n) in m.MPUWtuples:
            if not preparation:
                if n == place and t <= (1 - time) * m.s_T.at(-1):
                    getattr(m, label).add(expr=m.I[mtl,n,t] <= m.I_bnd[mtl,n,t][1] * off)
            else:
                if n == place and t >= (1 - time) * m.s_T.at(-1):
                    getattr(m, label).add(expr=m.I[mtl,n,t] <= m.I_bnd[mtl,n,t][1] * off)
    return m

def supply_disruption(m, supply, time, off, label, preparation=False):
    m.add_component(label, pyo.ConstraintList())

    for t in m.s_T:
        for (mtl, s) in m.MStuples:
            if not preparation:
                if s == supply and t <= (1 - time) * m.s_T.at(-1):
                    getattr(m, label).add(expr=m.B[mtl,s,t] <= m.B_bnd[mtl,s,t][1] * off)
            else:
                if s == supply and t >= (1 - time) * m.s_T.at(-1):
                    getattr(m, label).add(expr=m.B[mtl,s,t] <= m.B_bnd[mtl,s,t][1] * off)
    return m

