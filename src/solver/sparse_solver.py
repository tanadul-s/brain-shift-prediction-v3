from typing import Literal

import numpy as np
import scipy.sparse as spsp
import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpsp_linalg

from src.utils.loggerold import CustomLogger, TimeElapsedLog
from src.utils.logger import TaskLogger



class SparseSolver:
    def __init__(self, device: Literal["cpu", "gpu"] = "cpu"):
        self.MODE_CPU = False
        self.MODE_GPU = False
        if device == "cpu": self.MODE_CPU = True
        elif device == "gpu": self.MODE_GPU = True
        else: self.MODE_CPU = True
        
    def setup(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
        
    def solve(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    
class LUDecompositionSolver(SparseSolver):
    
    logger = TaskLogger("LU Decomposition Solver")

    def __init__(self, device: Literal["cpu", "gpu"] = "cpu"):
        super().__init__(device)
    
    @logger.logprocess("LU Decomposition")
    def setup(self, A: spsp.sparray):
        
        if not isinstance(A, spsp.csc_array): A = spsp.csc_array(A)
        
        if self.MODE_CPU:
            self.LU = spsp.linalg.splu(A)
        elif self.MODE_GPU:
            A_gpu = cpsp.csc_matrix(A)
            self.LU = cpsp_linalg.splu(A_gpu)
    
    def solve(self, b: np.ndarray):
        if self.MODE_CPU:
            x = self.LU.solve(b)
        elif self.MODE_GPU:
            b_gpu = cp.asarray(b)
            x_gpu = self.LU.solve(b_gpu)
            x = x_gpu.get()
        return x
    
    
class ConjugateGradientSolver(SparseSolver):

    logger = TaskLogger("Conjugate Gradient Solver")

    def __init__(self, preconditioner: Literal["jacobi"] = None, device: Literal["cpu", "gpu"] = "cpu"):
        super().__init__(device)
        self._preconditioner = preconditioner
    
    @logger.logprocess("Conjugate Gradient Setup")
    def setup(self, A: spsp.sparray):
        if not isinstance(A, spsp.csr_array): A = spsp.csr_array(A)
        
        if self.MODE_CPU: self.A = A
        elif self.MODE_GPU: self.A = cpsp.csr_matrix(A)
            
        if self._preconditioner == "jacobi":
            diag = self.A.diagonal()
            Minv = 1.0 / diag
            matvec = lambda x: Minv * x
            if self.MODE_CPU: self.M = spsp.linalg.LinearOperator(shape=self.A.shape, matvec=matvec)
            elif self.MODE_GPU: self.M = cpsp_linalg.LinearOperator(shape=self.A.shape, matvec=matvec)
        else:
            self.M = None
            
    def solve(self, b: np.ndarray):
        if self.MODE_CPU:
            x, _ = spsp.linalg.cg(self.A, b, M=self.M)
        elif self.MODE_GPU:
            b_gpu = cp.asarray(b)
            x_gpu, _ = cpsp_linalg.cg(self.A, b_gpu, M=self.M)
            x = x_gpu.get()
        return x