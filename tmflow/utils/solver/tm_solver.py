from dataclasses import dataclass, field
import numpy as np
import itertools as it
from numba import njit

import pickle

from tmflow.map import construct_tm_matrices
from tmflow.utils.weights import combine_weight_matrices


@dataclass
class TMSolver:
    order: int
    ode_dim: int
    ode_rhs: list
    map_step: float
    path: str = None
    calc_weights: bool = True
    symbolic_args: list = None
    numerical_args: list = None
    lexic_order: np.array = field(init=False)
    weights: np.array = field(init=False)

    def __post_init__(self):
        # get lexicographical order
        indices = []
        for j in range(self.order + 1):
            for jds in it.product(range(j + 1), repeat=self.ode_dim):
                if sum(jds) == j:
                    indices.append(jds)
        self.lexic_order = np.array(indices, dtype=np.int16)

        if self.calc_weights:
            # calculate weight matrices
            w = construct_tm_matrices(self.ode_rhs, self.ode_dim, self.order, self.map_step,
                                      sym_params=self.symbolic_args, num_params=self.numerical_args)
            self.weights = combine_weight_matrices(self.order, w)
        else:
            self.__load_matrices(self.path)

    def __load_matrices(self, path: str):
        """
        load  precomputed weights in pickle format to the solver.
        :param path: the path to the weights in pickle format
        """
        with open(path, 'rb') as weights_pkl:
            self.weights = pickle.load(weights_pkl)

    def torch_integrate(self, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Новый интерфейс, совместимый с torchdiffeq.odeint
        
        Parameters:
            y0: torch.Tensor shape (d,) или (b, d) для батча
            t: torch.Tensor shape (T,) временные точки
            
        Returns:
            torch.Tensor shape (T, d) или (b, T, d)
        """
        # Конвертируем torch -> numpy
        y0_np = y0.detach().cpu().numpy()
        t_np = t.detach().cpu().numpy()
        
        # Вычисляем решение (используем существующий метод)
        solution_np = run_tm_solver(y0_np, t_np, self)
        
        # Конвертируем обратно в torch
        solution = torch.from_numpy(solution_np).to(y0.device)
        
        # Добавляем поддержку батчей если нужно
        if y0.ndim == 2:
            solution = solution.permute(1, 0, 2)  # (batch, time, dim)
            
        return solution


@njit
def calc_numerical_solution(ini_data: list, time_grid: np.array, map_step: float, w_matrix: np.array, lexic: np.array,
                            dim: np.int16) -> np.array:
    """
    calculate ODEs numerical solution based on TM approch
    :param ini_data: initial values for ODEs
    :param time_grid: shape(N,) time grid where the solution should be computed
    :param map_step: Taylor map time step
    :param w_matrix: shape(dim,m) weights matrix united for all orders of nonlinearity
    :param lexic: shape(m,) order of indices for w_matrix
    :param dim: number of ODEs in the system
    :return: shape(N,dim) numerical solution computed in time_grid points
    """

    times_number = time_grid.shape[0]
    numerical_solution = np.zeros((times_number, dim), dtype=np.float64)
    numerical_solution[0, :] = ini_data
    cur_time = time_grid[0]
    inner_point = np.asarray(ini_data, dtype=np.float64)
    for i in range(1, times_number):
        while time_grid[i] > cur_time:
            cur_time += map_step
            inner_point = get_next_point(lexic, inner_point, w_matrix, dim)
        numerical_solution[i, :] = inner_point
    return numerical_solution


@njit
def get_next_point(lexic: np.array, prev_point: np.array, w_matrix: np.array, dim: np.int16) -> np.array:
    """
    compute the ODEs numerical solution in the next moment of time starting from prev_point
    :param lexic: shape(m,) order of indices for w_matrix
    :param prev_point: shape(dim,) previous point of the numerical solution
    :param w_matrix: shape(dim,m) weights matrix united for all orders of nonlinearity
    :param dim: number of ODEs in the system
    :return: shape(dim,) numerical solution computed in next moment of time
    """

    next_point = np.zeros(dim, dtype=np.float64)
    for i in range(dim):
        for idx, jds in enumerate(lexic):
            next_point[i] += w_matrix[i, idx] * np.prod(prev_point ** jds)
    return next_point


def run_tm_solver(ini_vals: list, time_grid: np.array, config: TMSolver) -> np.array:
    """
    run ODEs solver based on Taylor map
    :param ini_vals: initial values for ODEs
    :param time_grid
    :param config: configuration of solver
    :return: shape(N,dim) numerical solution computed in time_grid points
    """

    w_matrix = config.weights
    ini_data = np.asarray(ini_vals, dtype=np.float64)
    time_grid = np.asarray(time_grid, dtype=np.float64)
    w_matrix = np.asarray(w_matrix, dtype=np.float64)
    numerical_solution = calc_numerical_solution(ini_data, time_grid, config.map_step, w_matrix, config.lexic_order,
                                                 config.ode_dim)

    return numerical_solution

def odeint(func, y0, t, method='tmsolver', options=None):
    """
    Интерфейс, совместимый с torchdiffeq.odeint
    
    Parameters:
        func: Пока не используется, оставлен для совместимости
        y0: Начальные условия (torch.Tensor)
        t: Временные точки (torch.Tensor)
        options: Должен содержать параметры для TMSolver:
            - order: порядок метода
            - ode_rhs: правые части ОДУ
            - map_step: шаг
            - ode_dim: размерность системы
    """
    if options is None:
        options = {}
    
    solver = TMSolver(
        order=options.get('order', 4),
        ode_dim=options['ode_dim'],
        ode_rhs=options['ode_rhs'],
        map_step=options['map_step'],
        calc_weights=True
    )
    
    return solver.torch_integrate(y0, t)
