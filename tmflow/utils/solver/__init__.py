from .tm_solver import (
    TMSolver,
    run_tm_solver,
    calc_numerical_solution,
    get_next_point,
    odeint  # обязательно добавьте это
)

__all__ = [
    'TMSolver',
    'run_tm_solver',
    'calc_numerical_solution',
    'get_next_point',
    'odeint'  # и здесь
]
