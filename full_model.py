import pyomo.environ as pyo
from data_instances import data_case_study
from utils import data_preprocessing, solve_mip, print_summary, result_aggregation
from disruptions import plant_disruption, arc_disruption, inventory_disruption, supply_disruption

def full_supply_chain(data, time_production=False, fixed_transport=False, final_deviation=False, inventory_deviation=False, sla_flag=None):
    # Create model
    m = pyo.ConcreteModel()

    # Append flags
    m.time_production = time_production
    m.fixed_transport = fixed_transport
    m.final_deviation = final_deviation
    m.inventory_deviation = inventory_deviation
    m.sla_flag = sla_flag
    
    # Declare Sets
    m.s_S = pyo.Set(initialize=data['S'])
    m.s_P = pyo.Set(initialize=data['P'])
    m.s_W = pyo.Set(initialize=data['W'])
    m.s_C = pyo.Set(initialize=data['C'])
    m.s_A = pyo.Set(initialize=data['A'])
    m.s_M = pyo.Set(initialize=data['M'])
    m.s_T = pyo.Set(initialize=data['T'])

    m.s_N = pyo.Set(initialize=data['N'])
    m.s_PUW = pyo.Set(initialize=data['PUW'])

    # Declare Set tuples
    m.MStuples = pyo.Set(initialize=data['MStuples'])
    m.MPtuples = pyo.Set(initialize=data['MPtuples'])
    m.MWtuples = pyo.Set(initialize=data['MWtuples'])
    m.MCtuples = pyo.Set(initialize=data['MCtuples'])
    m.MAtuples = pyo.Set(initialize=data['MAtuples'])
    m.PRtuples = pyo.Set(initialize=data['PRtuples'])
    m.MPUWtuples = pyo.Set(initialize=data['MPUWtuples'])

    m.s_R = pyo.Param(m.s_P, initialize=data['R'], within=pyo.Any)
    m.A_in = pyo.Param(m.s_N, initialize=data['A_in'], within=pyo.Any)
    m.A_out = pyo.Param(m.s_N, initialize=data['A_out'], within=pyo.Any)

    # Declare Parameters
    m.P_bnd = pyo.Param(m.PRtuples, m.s_T, initialize=data['P_bnd'], within=pyo.Any)
    m.I_bnd = pyo.Param(m.MPUWtuples, m.s_T, initialize=data['I_bnd'], within=pyo.Any)
    m.B_bnd = pyo.Param(m.MStuples, m.s_T, initialize=data['B_bnd'], within=pyo.Any)
    m.F_bnd = pyo.Param(m.MAtuples, m.s_T, initialize=data['F_bnd'], within=pyo.Any)

    m.delta = pyo.Param(m.MCtuples, m.s_T, initialize=data['delta'], within=pyo.NonNegativeReals)
    m.tau = pyo.Param(m.MAtuples, m.s_T, initialize=data['tau'], within=pyo.NonNegativeIntegers) 
    m.I0 = pyo.Param(m.MPUWtuples, initialize=data['I0'], within=pyo.NonNegativeReals)
    m.phi = data['phi'] # Creative license            

    m.lamC = pyo.Param(m.MCtuples, m.s_T, initialize=data['lamC'], within=pyo.NonNegativeReals)
    m.lamF = pyo.Param(m.MAtuples, m.s_T, initialize=data['lamF'], within=pyo.NonNegativeReals)
    m.lamB = pyo.Param(m.MStuples, m.s_T, initialize=data['lamB'], within=pyo.NonNegativeReals)
    m.lamP = pyo.Param(m.PRtuples, m.s_T, initialize=data['lamP'], within=pyo.NonNegativeReals)
    m.lamU = pyo.Param(m.MCtuples, m.s_T, initialize=data['lamU'], within=pyo.NonNegativeReals, mutable=True)
    m.lamI = pyo.Param(m.MPUWtuples, m.s_T, initialize=data['lamI'], within=pyo.NonNegativeReals)
    m.lamdelta = pyo.Param(m.MCtuples, m.s_T, initialize=data['lamdelta'], within=pyo.NonNegativeReals, mutable=True)

    # Declare Variables
    m.B = pyo.Var(m.MStuples, m.s_T, initialize=0, bounds=lambda _,mtl,s,t: m.B_bnd[mtl,s,t], within=pyo.NonNegativeReals)
    m.I = pyo.Var(m.MPUWtuples, m.s_T, bounds=lambda _,mtl,n,t: (0,m.I_bnd[mtl,n,t][1]), within=pyo.NonNegativeReals)
    m.Fin = pyo.Var(m.MAtuples, m.s_T, initialize=0, bounds=lambda _,mtl,nin,nout,tp,t: m.F_bnd[mtl,nin,nout,tp,t], within=pyo.NonNegativeReals)
    m.Fout = pyo.Var(m.MAtuples, m.s_T, initialize=0, bounds=lambda _,mtl,nin,nout,tp,t: m.F_bnd[mtl,nin,nout,tp,t], within=pyo.NonNegativeReals)
    m.D = pyo.Var(m.MCtuples, m.s_T, initialize=0, bounds=(0,None), within=pyo.NonNegativeReals )
    m.U = pyo.Var(m.MCtuples, m.s_T, initialize=0, bounds=(0,None), within=pyo.NonNegativeReals )
    m.y = pyo.Var(m.MCtuples, m.s_T, within=pyo.Binary)

    # Declare variables for time dependency extension
    if not time_production:
        m.P = pyo.Var(m.PRtuples, m.s_T, initialize=0, bounds=lambda _,p,r,t: m.P_bnd[p,r,t], within=pyo.NonNegativeReals)
    else:
        m.Pin = pyo.Var(m.PRtuples, m.s_T, initialize=0, bounds=lambda _,p,r,t: m.P_bnd[p,r,t], within=pyo.NonNegativeReals)
        m.Pout = pyo.Var(m.PRtuples, m.s_T, initialize=0, bounds=lambda _,p,r,t: m.P_bnd[p,r,t], within=pyo.NonNegativeReals)
        m.tauP = pyo.Param(m.PRtuples, m.s_T, initialize=data['tauP'], within=pyo.NonNegativeIntegers)

    # Fix non existent orders
    for t in m.s_T:
        for (mtl,c) in m.MCtuples:
            if m.delta[mtl,c,t] == 0:
                m.y[mtl,c,t].fix(0)

    # Declare Constraints
    @m.Constraint(m.MAtuples, m.s_T)
    def transport_delay(m, mtl, nin, nout, tp, t):
        if t + m.tau[mtl, nin, nout, tp, t] <= m.s_T.at(-1):
            return m.Fin[mtl, nin, nout, tp, t] == m.Fout[mtl, nin, nout, tp, t+m.tau[mtl, nin, nout, tp, t]]
        return pyo.Constraint.Skip
    
    # Fix instead of constraint for efficency in solve
    for (mtl, nin, nout, tp) in m.MAtuples:
        m.Fin[mtl, nin, nout, tp, 0].fix(0)
        m.Fout[mtl, nin, nout, tp, 0].fix(0)
        for t in m.s_T:
            if t + m.tau[mtl, nin, nout, tp, t] > m.s_T.at(-1):
                m.Fin[mtl, nin, nout, tp, t].fix(0)
            if t - m.tau[mtl, nin, nout, tp, t] < 0:
                m.Fout[mtl, nin, nout, tp, t].fix(0)

    @m.Constraint(m.MWtuples, m.s_T)
    def warehouse_balance(m, mtl, w, t):
        if t == 0:
            return m.I[mtl, w, t] == m.I0[mtl, w]
        return m.I[mtl, w, t] == m.I[mtl, w, t-1] + sum(m.Fout[mtl, nin, nout, tp, t] for (nin, nout, tp) in m.A_in[w]  if (mtl, (nin, nout, tp)) in m.MAtuples) \
            - sum(m.Fin[mtl, nin, nout, tp, t] for (nin, nout, tp) in m.A_out[w]  if (mtl, (nin, nout, tp)) in m.MAtuples ) 

    @m.Constraint(m.MCtuples, m.s_T)
    def demand_calculation(m, mtl, c, t):
        if t == 0:  
            return m.D[mtl, c, t] == 0
        return m.D[mtl, c, t] == sum(m.Fout[mtl, nin, nout, tp, t] for (nin, nout, tp) in m.A_in[c]  if (mtl, (nin, nout, tp)) in m.MAtuples )
    
    @m.Constraint(m.MStuples, m.s_T)
    def supply_calculation(m, mtl, s, t):
        if t == 0:  
            return m.B[mtl, s, t] == 0
        return m.B[mtl,s,t] == sum( m.Fin[mtl, nin, nout, tp, t] for (nin,nout,tp) in m.A_out[s] if (mtl, (nin, nout, tp)) in m.MAtuples )
    
    
    @m.Constraint(m.MCtuples, m.s_T)
    def unmet_demand_balance(m, mtl ,c, t):
        if t == 0:
            return m.U[mtl, c, t] == 0   #TODO: Check if needed
        return m.U[mtl, c, t] == m.U[mtl, c, t-1] - m.D[mtl, c, t] + m.delta[mtl, c, t]*(1 - m.y[mtl,c,t])
    
    # Fixed transport extension
    if fixed_transport:
        m.lamFfix = pyo.Param(m.MAtuples, m.s_T, initialize=data['lamFfix'], within=pyo.NonNegativeReals)
        m.x = pyo.Var(m.MAtuples, m.s_T, within=pyo.Binary)

        @m.Constraint(m.MAtuples, m.s_T)
        def fixed_transport_lb(m , mtl, nin, nout, tp, t):
            return m.F_bnd[mtl, nin, nout, tp, t][0]*m.x[mtl, nin, nout, tp, t] <= m.Fin[mtl, nin, nout, tp, t]

        @m.Constraint(m.MAtuples, m.s_T)
        def fixed_transport_ub(m , mtl, nin, nout, tp, t):
            return m.F_bnd[mtl, nin, nout, tp, t][1]*m.x[mtl, nin, nout, tp, t] >= m.Fin[mtl, nin, nout, tp, t]
        
    # Final time inventory deviation extension
    if not final_deviation:
        @m.Constraint(m.MPUWtuples)
        def final_time(m, mtl, n):
            return m.I[mtl, n, 0] == m.I[mtl, n, m.s_T.at(-1)]
        
    else:
        m.deviation = pyo.Var(m.MPUWtuples, initialize=0, within=pyo.NonNegativeReals)
        m.lamdev = pyo.Param(m.MPUWtuples, initialize=data['lamdev'], within=pyo.NonNegativeReals)

        @m.Constraint(m.MPUWtuples)
        def final_inventory_dev_right(m, mtl, n):
            return m.deviation[mtl, n] >= m.I0[mtl, n] - m.I[mtl, n, m.s_T.at(-1)]
        
        @m.Constraint(m.MPUWtuples)
        def final_inventory_dev_left(m, mtl, n):
            return m.deviation[mtl, n] >= m.I[mtl, n, m.s_T.at(-1)] - m.I0[mtl, n]
        

    # Time dependent production
    if not time_production:
        for (p, r) in m.PRtuples:
            m.P[p, r, 0].fix(0)

        @m.Constraint(m.MPtuples, m.s_T)
        def plant_balance(m, mtl, p, t):
            if t == 0:
                return m.I[mtl, p, t] == m.I0[mtl, p]
            return m.I[mtl, p, t] == m.I[mtl, p, t-1] \
                + sum( m.Fout[mtl, nin, nout, tp, t] for (nin,nout,tp) in m.A_in[p] if (mtl, (nin, nout, tp)) in m.MAtuples) \
                - sum( m.Fin[mtl, nin, nout, tp, t] for (nin,nout,tp) in m.A_out[p]  if (mtl, (nin, nout, tp)) in m.MAtuples ) \
                + sum( m.phi[r,mtl]*m.P[p,r,t] for r in m.s_R[p] )
    else:
        for (p, r) in m.PRtuples:
            m.Pin[p, r, 0].fix(0)
            m.Pout[p, r, 0].fix(0)
            for t in m.s_T:
                if t + m.tauP[p, r, t] > m.s_T.at(-1):
                    m.Pin[p, r, t].fix(0)
                if t - m.tauP[p, r, t] < 0:
                    m.Pout[p, r, t].fix(0)

        @m.Constraint(m.PRtuples, m.s_T)
        def production_delay(m, p, r, t):
            if t + m.tauP[p, r, t] <= m.s_T.at(-1):
                return m.Pin[p, r, t] == m.Pout[p, r, t+m.tauP[p, r, t]]
            return pyo.Constraint.Skip
        
        @m.Constraint(m.MPtuples, m.s_T)
        def plant_balance_time(m, mtl, p, t):
            if t == 0:
                return m.I[mtl, p, t] == m.I0[mtl, p]
            return m.I[mtl, p, t] == m.I[mtl, p, t-1] \
                + sum( m.Fout[mtl, nin, nout, tp, t] for (nin,nout,tp) in m.A_in[p] if (mtl, (nin, nout, tp)) in m.MAtuples) \
                - sum( m.Fin[mtl, nin, nout, tp, t] for (nin,nout,tp) in m.A_out[p]  if (mtl, (nin, nout, tp)) in m.MAtuples ) \
                + sum( m.phi[r,mtl]*m.Pout[p,r,t] for r in m.s_R[p] if m.phi[r,mtl] > 0) \
                + sum( m.phi[r,mtl]*m.Pin[p,r,t] for r in m.s_R[p] if m.phi[r,mtl] < 0)

    # Negative inventory deviation extension
    if not inventory_deviation:
        for (mtl,n) in m.MPUWtuples:
            for t in m.s_T:
                m.I[mtl,n,t].lower = m.I_bnd[mtl,n,t][0]
    else:
        m.z = pyo.Var(m.MPUWtuples, m.s_T, within=pyo.Binary)
        m.K = pyo.Var(m.MPUWtuples, m.s_T, bounds=lambda _,mtl,n,t: (-m.I_bnd[mtl,n,t][0], 0), within=pyo.NonPositiveReals)
        m.lamK = pyo.Param(m.MPUWtuples, m.s_T, initialize=data['lamK'], within=pyo.NonNegativeReals)

        @m.Constraint(m.MPUWtuples, m.s_T)
        def relu1(m, mtl, n, t):
            return m.K[mtl,n,t] <=  m.I[mtl,n,t] - m.I_bnd[mtl,n,t][0]
        
        @m.Constraint(m.MPUWtuples, m.s_T)
        def relu2(m, mtl, n, t):
            return m.K[mtl,n,t] >= m.I[mtl,n,t] - m.I_bnd[mtl,n,t][0] + (m.I_bnd[mtl,n,t][0] - m.I_bnd[mtl,n,t][1]) * (1 - m.z[mtl,n,t])
        
        @m.Constraint(m.MPUWtuples, m.s_T)
        def relu3(m, mtl, n, t):
            return m.K[mtl,n,t] >= - m.I_bnd[mtl,n,t][0] * m.z[mtl,n,t]
        
    # Service level agreements
    if sla_flag:

        m.MS_SLAtuples = pyo.Set(initialize=data['MS_SLAtuples'])
        m.B_SLA = pyo.Param(m.MS_SLAtuples, m.s_T, initialize=data['B_SLA'], within=pyo.NonNegativeReals)
        m.w = pyo.Var(m.MS_SLAtuples, m.s_T, within=pyo.Binary)

        if sla_flag == 'simple':
            @m.Constraint(m.MS_SLAtuples, m.s_T)
            def sla_simple_lb(m , mtl, s, t):
                return m.B_SLA[mtl, s, t]*m.w[mtl, s, t] <= m.B[mtl, s, t]

            @m.Constraint(m.MS_SLAtuples, m.s_T)
            def sla_simple_ub(m , mtl, s, t):
                return m.B_bnd[mtl, s, t][1]*m.w[mtl, s, t] >= m.B[mtl, s, t]
            
        elif sla_flag == 'window':
            m.tauSLA = pyo.Param(m.MS_SLAtuples, m.s_T, initialize=data['tauSLA'], within=pyo.NonNegativeIntegers)

            @m.Constraint(m.MS_SLAtuples, m.s_T)
            def sla_window_lb(m , mtl, s, t):
                if t + m.tauSLA[mtl, s, t] <= m.s_T.at(-1):
                    return m.B_SLA[mtl, s, t]*m.w[mtl, s, t] <= sum(m.B[mtl, s, tt] for tt in range(t, t + m.tauSLA[mtl, s, t] + 1)) 
                return pyo.Constraint.Skip

            @m.Constraint(m.MS_SLAtuples, m.s_T)
            def sla_window_ub(m , mtl, s, t):
                if t + m.tauSLA[mtl, s, t] <= m.s_T.at(-1):
                    return m.B_bnd[mtl, s, t][1]*m.w[mtl, s, t] >= sum(m.B[mtl, s, tt] for tt in range(t, t + m.tauSLA[mtl, s, t] + 1)) 
                return pyo.Constraint.Skip 
        else:
            print('Enter a valid flag for SLA ("", "simple" or "window")')


    # Objective
    @m.Objective(sense = pyo.maximize)
    def obj(m):
        objective = sum( sum(m.lamC[mtl,c,t]*m.D[mtl,c,t] for (mtl,c) in m.MCtuples)\
            - sum(m.lamF[mtl,nin,nout,tp,t]*m.Fout[mtl,nin,nout,tp,t] for (mtl,nin,nout,tp) in m.MAtuples)\
            - sum(m.lamB[mtl,s,t]*m.B[mtl,s,t] for (mtl,s) in m.MStuples)\
            - sum(m.lamU[mtl,c,t]*m.U[mtl,c,t] for (mtl,c) in m.MCtuples)\
            - sum(m.lamI[mtl,n,t]*m.I[mtl,n,t] for (mtl,n) in m.MPUWtuples)\
            - sum(m.lamdelta[mtl,c,t]*m.y[mtl,c,t] for (mtl,c) in m.MCtuples) for t in m.s_T)
        
        if not time_production:
            objective -= sum(m.lamP[p,r,t]*m.P[p,r,t] for (p,r) in m.PRtuples for t in m.s_T)
        else:
            objective -= sum(m.lamP[p,r,t]*m.Pin[p,r,t] for (p,r) in m.PRtuples for t in m.s_T)
        
        if fixed_transport:
            objective -= sum(m.lamFfix[mtl,nin,nout,tp,t]*m.x[mtl,nin,nout,tp,t] for (mtl,nin,nout,tp) in m.MAtuples for t in m.s_T)

        if final_deviation:
            objective -= sum(m.lamdev[mtl,n]*m.deviation[mtl,n] for (mtl,n) in m.MPUWtuples)

        if inventory_deviation:
            objective += sum(m.lamK[mtl,n,t]*m.K[mtl,n,t] for (mtl,n) in m.MPUWtuples for t in m.s_T)
        
        return objective
    
    
    return m

if __name__ == '__main__':

    data = data_case_study(TF=120, random_order_time=False, random_cost=False)
    p_data = data_preprocessing(data, alpha=0)
    m = full_supply_chain(p_data, time_production=False, fixed_transport=False, final_deviation=True, inventory_deviation=False, sla_flag=None)

    # m = plant_disruption(m, plant='P-Plant1', time=0.5, off=0.5, label='plant_dis_1') 
    # m = arc_disruption(m, arc=('P-Plant1', 'W-Warehouse1', 'Truck'), time=0.5, off=0, label='arc_truck_dis')
    # m = inventory_disruption(m, place='W-Warehouse1', time=0.0, off=0.15, label='warehouse_dis')
    # m = supply_disruption(m, supply='S-Supplier1', time=0, off=0.1, label='supply_dis')

    m = solve_mip(m, solvername='gurobi', time_limit=1000, abs_gap=0.005, threads=8, tee=True)

    print_summary(m, size=False, variables=True, eps=1e-3)

    

