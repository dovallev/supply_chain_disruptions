import pyomo.environ as pyo
import math
import cloudpickle as cp
import numpy as np
import os


def data_preprocessing(d, alpha=0):
    d['N'] = d['S'] + d['P'] + d['W'] + d['C']
    d['PUW'] = d['P'] + d['W']

    A_in, A_out = dict(), dict()

    for n in d['N']:
        an_in = list()
        an_out = list()
        for a in d['A']:
            if n == a[0]:
                an_out.append(a)
            elif n == a[1]:
                an_in.append(a)
        A_in[n] = an_in
        A_out[n] = an_out

    d['A_in'] = A_in
    d['A_out'] = A_out

    for (m, n) in d['MPUWtuples']:
        for t in d['T']:
            d['I_bnd'][m,n,t] = (np.minimum(alpha*d['SS'][m,n,t], d['I0'][m,n]), d['I_bnd'][m,n,t][1])

    return d


def solve_mip(m, solvername='gurobi', time_limit=100, abs_gap=0.0, threads=8, tee=True, dual_flag=False):
    if solvername == 'gurobi':
        opt = pyo.SolverFactory('gurobi')
        opt.options['TimeLimit'] = time_limit
        opt.options['MIPGap'] = abs_gap
        opt.options['Threads'] = threads
        if dual_flag:
            opt.options['DualReductions'] = 0
        m.results = opt.solve(m, tee=tee)

    else:
        # SOLVE
        opt = pyo.SolverFactory('gams', solver=solvername)
        m.results = opt.solve(m, tee=tee,
                            add_options=[
                                'option reslim = ',str(time_limit),';'
                                'option optcr = ',str(abs_gap),';'])

    return m

def result_aggregation(m, eps=1e-5):
    buy_c, fin_c, fout_c, prod_c, invw_c, invp_c, dem_c, unm_c, fdev_c, pout_c, negi_c = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    buy_q, fin_q, fout_q, prod_q, invw_q, invp_q, dem_q, unm_q, fdev_q, pout_q, negi_q = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cancel, slas, = 0, 0

    for t in m.s_T:
        for (mtl, s) in m.MStuples:
            if m.B[mtl,s,t]() > eps:
                buy_c += 1
                buy_q += m.B[mtl,s,t]()
        
        for (mtl, nin, nout, tp) in m.MAtuples:
            if m.Fin[mtl,nin,nout,tp,t]() > eps:
                fin_c += 1
                fin_q += m.Fin[mtl,nin,nout,tp,t]()

            if m.Fout[mtl,nin,nout,tp,t]() > eps:
                fout_c += 1
                fout_q += m.Fout[mtl,nin,nout,tp,t]()

        for (mtl, n) in m.MWtuples:
            if t > 0:
                if not math.isclose(m.I[mtl,n,t](),m.I[mtl,n,t-1](), rel_tol=eps):
                    invw_c += 1
                    invw_q += m.I[mtl,n,t]()

        for (mtl, n) in m.MPtuples:
            if t > 0:
                if not math.isclose(m.I[mtl,n,t](),m.I[mtl,n,t-1](), rel_tol=eps):
                    invp_c += 1
                    invp_q += m.I[mtl,n,t]()
        
        for (mtl, c) in m.MCtuples:
            if m.D[mtl,c,t]() > eps:
                dem_c += 1
                dem_q += m.D[mtl,c,t]()

            if m.U[mtl,c,t]() > eps:
                unm_c += 1
                unm_q += m.U[mtl,c,t]()

            if m.y[mtl,c,t]() > 5*eps:
                cancel += 1
    
    assert  fin_c == fout_c

    if not m.time_production:
        for (p, r) in m.PRtuples:
            for t in m.s_T:
                if m.P[p,r,t]() > eps:
                    prod_c += 1
                    prod_q += m.P[p,r,t]()
    else:
        for (p, r) in m.PRtuples:
            for t in m.s_T:
                if m.Pin[p,r,t]() > eps:
                    prod_c += 1
                    prod_q += m.Pin[p,r,t]()

                if m.Pout[p,r,t]() > eps:
                    pout_c += 1
                    pout_q += m.Pout[p,r,t]()

        assert prod_c == pout_c
        assert prod_q == pout_q

    if m.final_deviation:
        for (mtl, n) in m.MPUWtuples:
            if m.deviation[mtl, n]() > eps:
                fdev_c += 1
                fdev_q += m.deviation[mtl, n]()

    if m.inventory_deviation:
        for t in m.s_T:
            for (mtl, n) in m.MPUWtuples:
                if m.K[mtl, n, t]() < -eps:
                    negi_c += 1
                    negi_q += m.K[mtl, n, t]()

    if m.sla_flag:
        for t in m.s_T:
            for (mtl, s) in m.MS_SLAtuples:
                if m.w[mtl, s, t]() is not None:
                    if m.w[mtl, s, t]() > 5*eps :
                        slas += 1
    
    counters = {'buy_c':buy_c, 'fin_c':fin_c, 'fout_c':fout_c, 'prod_c':prod_c, 'invw_c':invw_c, 'invp_c':invp_c, 'dem_c':dem_c, 'unm_c':unm_c, 'fdev_c':fdev_c, 'pout_c':pout_c, 'negi_c':negi_c}
    quantity = {'buy_q':buy_q, 'fin_q':fin_q, 'fout_q':fout_q, 'prod_q':prod_q, 'invw_q':invw_q, 'invp_q':invp_q, 'dem_q':dem_q, 'unm_q':unm_q, 'fdev_q':fdev_q, 'pout_q':pout_q, 'negi_q':negi_q}
    binaries = {'cancel':cancel, 'slas':slas}

    return counters, quantity, binaries

def print_summary(m, size=True, variables=True, eps=1e-5):
    print('_____________________________')
    print('Objective:', round(pyo.value(m.obj), 5))
    print('Termination:', str(m.results.solver.termination_condition))
    print('Time:', str(round(float(m.results.solver.wall_time), 5)))

    if size:
        print('_____________________________')
        print('Continuous Vars:', m.results.problem.number_of_continuous_variables)
        print('Binary Vars:', m.results.problem.number_of_binary_variables)
        print('Constraints:', m.results.problem.number_of_constraints)
    
    if variables:
        c, q, b = result_aggregation(m)
        
        print('_____________________________')
        print('Buy:', c['buy_c'])
        print('Flow:', c['fin_c'])
        print('Inventory W:', c['invw_c'])
        print('Inventory P:', c['invp_c'])
        print('Demand:', c['dem_c'])
        print('Produce:', c['prod_c'])

        if m.final_deviation:
            print('Final Deviations:', c['fdev_c'])

        if m.inventory_deviation:
            print('Underpassing:', c['negi_c'])

        if m.sla_flag:
            print('SLAs:', b['slas'])

        print('Late:', c['unm_c'])
        print('Cancel:', b['cancel'])

    print('_____________________________')

def to_pickle(object, path): 
    current_path = os.path.dirname(os.path.realpath(__file__))
    unsolved_model_pickle_path = os.path.join(current_path, path)
    with open(unsolved_model_pickle_path, "wb") as f:
        cp.dump(object, f)
    return object

def from_pickle(path):
    # current_path = os.getcwd()
    current_path = os.path.dirname(os.path.realpath(__file__))
    unsolved_model_pickle_path = os.path.join(current_path, path)
    with open(unsolved_model_pickle_path, "rb") as f:
        object = cp.load(f)
    return object
    

 

    

                
