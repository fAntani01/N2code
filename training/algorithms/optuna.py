from typing import Callable, Optional
import optuna
from optuna.samplers import TPESampler


class OptunaOptimizer:


    def __init__(self, objective_fn: Callable[[list[float], Optional[str]], float], x0: list[float], study_path: str, startup_trials: int = 20, min_value: float = 0.0, max_value: float = 60.0, step: float = 5.0):
          
        
        self.objective_fn = objective_fn
        self.x = list(x0)
        self.min_value = min_value
        self.max_value = max_value
        self.step = step

        sampler = TPESampler(seed=42, n_startup_trials=startup_trials, multivariate=True) # Modify as needed
        self.study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{study_path}", sampler=sampler, direction="maximize", load_if_exists=True)


    def step(self) -> tuple[list[float], float, dict]:

        # Ask a new trial
        trial = self.study.ask()

        # Sample all parameters
        new_values = []
        for i in range(len(self.x)):
            param_name = f"param_{i}"
            value = trial.suggest_float(param_name, self.min_value, self.max_value, step=self.step)
            new_values.append(value)

        # Evaluate the objective function
        score = self.objective_fn(new_values)

        # Tell the study about the result
        self.study.tell(trial, score)


        return new_values, score, {}

