# %%
from pyomo.environ import *
import numpy as np


# %%
# The deterministic model specified 
def get_deterministic_model(coef: np.array, err: float):
    
    # Workaround with BUG https://github.com/Pyomo/pyomo/issues/31
    coef_ = coef.tolist()
    
    model = ConcreteModel()

    model.x1 = Var(within=NonNegativeReals)
    model.x2 = Var(within=NonNegativeReals)
    
    model.BoundLimits11 = Constraint(expr = model.x1 >= 0.7)
    model.BoundLimits12 = Constraint(expr = model.x1 <= 296.4)
    model.BoundLimits22 = Constraint(expr = model.x2 <= 49.6)
    
    
    model.AdsLimit = Constraint(
        expr = model.x1 + model.x2 <= 200
    )
    
    model.PolicyConstraint = Constraint(
        expr = model.x1 - 0.5 * model.x2 >= 0
    )
    
    model.y_a = Var(within=NonNegativeReals, bounds=(None, 8))
    model.y_b = Var(within=NonNegativeReals, bounds=(None, 12))

    model.CapacityConstraint = Constraint(
        expr = 3*model.y_a + 2*model.y_b <= 36
    )
    
    model.SaleConstraint = Constraint(
        expr = model.y_a + model.y_b <= coef_[0] + coef_[1] * model.x1 + coef_[2] * model.x2 + err
    )

    model.Profit = Expression(expr=3*model.y_a+5*model.y_b)
    model.obj = Objective(expr=0.1*model.x1 + 0.5*model.x2 - model.Profit, sense=minimize)
    
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT) 

    return model


# %%
# LEO-Wyndor-2020 model.
# We simply grab the LEO-Wyndor model and patch the changed specifications.
def get_deterministic_model_2020(coef: np.array, err: np.array):
    model = get_deterministic_model(coef, err)
    
    # Patch the model objective
    del model.obj
    model.obj = Objective(expr=0.5*model.x1 + 0.2*model.x2 - model.Profit, sense=minimize)
    
    # Patch the model policy change
    del model.PolicyConstraint
    
    model.PolicyConstraint1 = Constraint(
        expr = 0.2*model.x2 <= model.x1
    )
    
    model.PolicyConstraint2 = Constraint(
        expr = model.x1 <= 0.7*model.x2
    )
    
    return model


# %%
def get_all_in_one_model(coef: np.array, errs: np.array):
    N = len(errs)
    index_set = list(range(N))
    
    coef_ = coef.tolist()
    errs_ = errs.tolist()
    
    model = ConcreteModel()

    model.x1 = Var(within=NonNegativeReals, bounds=(0.7, 296.4))
    model.x2 = Var(within=NonNegativeReals, bounds=(0, 49.6))
        
    model.AdsLimit = Constraint(
        expr = model.x1 + model.x2 <= 200
    )
    
    model.PolicyConstraint = Constraint(
        expr = model.x1 - 0.5 * model.x2 >= 0
    )
    
    model.y_a = Var(index_set, within=NonNegativeReals, bounds=(None, 8))
    model.y_b = Var(index_set, within=NonNegativeReals, bounds=(None, 12))

    def CapacityConstraintRule(model, i):
        return 3*model.y_a[i] + 2*model.y_b[i] <= 36
    
    model.CapacityConstraint = Constraint(
        index_set, rule=CapacityConstraintRule
    )
    
    def SaleConstraintRule(model, i):
        return (
            model.y_a[i] + model.y_b[i]
            <= coef_[0] + coef_[1] * model.x1 + coef_[2] * model.x2 + errs_[i]
        )
    
    model.SaleConstraint = Constraint(
        index_set, rule=SaleConstraintRule
    )
    
    model.obj = Objective(
        expr= 0.1*model.x1 + 0.5*model.x2
        - (1/N)*sum(3*model.y_a[i] + 5*model.y_b[i]
                    for i in index_set)
    )

    return model


# %%
# Returns a second stage model 2a-2f with given error
def get_second_stage_model(x: np.array, coef: np.array, err: float):
    
    coef_ = coef.tolist()
    x1, x2 = x.tolist()
    
    model = ConcreteModel()
    
    model.y_a = Var(within=NonNegativeReals)
    model.y_b = Var(within=NonNegativeReals)

    model.ProductionConstraint1 = Constraint(
        expr = model.y_a <= 8
    )

    model.ProductionConstraint2 = Constraint(
        expr = 2*model.y_b <= 24
    )

    model.CapacityConstraint = Constraint(
        expr = 3*model.y_a + 2*model.y_b <= 36
    )
    
    model.SaleConstraint = Constraint(
        expr = model.y_a + model.y_b <= coef_[0] + coef_[1] * x1 + coef_[2] * x2 + err
    )

    model.Profit = Expression(expr=3*model.y_a+5*model.y_b)
    
    model.obj = Objective(expr=-model.Profit, sense=minimize)
    return model


# %%
# Piecewise model for LEO-Wyndor-2018
# coef should be array of size (P, d)
# where P is the number of pieces
# (:, 0) constant (:, 1) TV (:, 2) Radio
def get_piecewise_model(coef: np.array, errs: np.array):
    
    N = len(errs)
    
    coef_ = coef.tolist()
    errs_ = errs.tolist()
    
    # The index sets
    index_set = list(range(N))
    index_piece = list(range(len(coef_)))
    
    model = ConcreteModel()

    model.x1 = Var(within=NonNegativeReals, bounds=(0.7, 296.4))
    model.x2 = Var(within=NonNegativeReals, bounds=(0, 49.6))
        
    model.AdsLimit = Constraint(
        expr = model.x1 + model.x2 <= 200
    )
    
    model.PolicyConstraint = Constraint(
        expr = model.x1 - 0.5 * model.x2 >= 0
    )
    
    model.y_a = Var(index_set, within=NonNegativeReals, bounds=(None, 8))
    model.y_b = Var(index_set, within=NonNegativeReals, bounds=(None, 12))
    model.t = Var(within=NonNegativeReals)
    
    def TConstraintRule(model, ip):
        return model.t <= coef_[ip][0] + coef_[ip][1] * model.x1 + coef_[ip][2] * model.x2
    model.TConstraint = Constraint(
        index_piece, rule=TConstraintRule
    )
    
    def CapacityConstraintRule(model, i):
        return 3*model.y_a[i] + 2*model.y_b[i] <= 36
    
    model.CapacityConstraint = Constraint(
        index_set, rule=CapacityConstraintRule
    )

    def SaleConstraintRule(model, i):
        return (
            model.y_a[i] + model.y_b[i]
            <= model.t + errs_[i]
        )
    
    model.SaleConstraint = Constraint(
        index_set, rule=SaleConstraintRule
    )
    
    model.obj = Objective(
        expr= 0.1*model.x1 + 0.5*model.x2
        - (1/N)*sum(3*model.y_a[i] + 5*model.y_b[i]
                    for i in index_set)
    )

    return model


# %%
# Returns a second stage model 2a-2f with given error
def get_second_stage_piecewise_model(x: np.array, coef: np.array, err: float):
    
    coef_ = coef.tolist()
    x1, x2 = x.tolist()
        
    model = ConcreteModel()
    
    model.y_a = Var(within=NonNegativeReals)
    model.y_b = Var(within=NonNegativeReals)

    model.ProductionConstraint1 = Constraint(
        expr = model.y_a <= 8
    )

    model.ProductionConstraint2 = Constraint(
        expr = 2*model.y_b <= 24
    )

    model.CapacityConstraint = Constraint(
        expr = 3*model.y_a + 2*model.y_b <= 36
    )
    
    # The piecewise sales constraint
    index_piece = list(range(len(coef_)))
    def SaleConstraintRule(model, i):
        return (
            model.y_a + model.y_b <= coef_[i][0] + float(coef_[i][1]) * x1 + float(coef_[i][2]) * x2 + err
        )
    
    model.SaleConstraint = Constraint(index_piece, rule=SaleConstraintRule)

    model.Profit = Expression(expr=3*model.y_a+5*model.y_b)
    
    model.obj = Objective(expr=-model.Profit, sense=minimize)
    return model

