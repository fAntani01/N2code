# Ogni algoritmo definisce la propria lista di parametri liberi.
# "Iterations" e' comune a tutti e non e' in questo dizionario, perche'
# resta fissa nel .ui (train_input_n_iter_2).
#
# key    -> nome del parametro usato per costruire l'optimizer
# label  -> testo mostrato nella QLabel
# default-> testo di default messo nella QLineEdit
# parser -> funzione per convertire il testo letto in un valore utilizzabile
ALGORITHM_PARAMS = {
    "Direct Search": [{"key": "values_set", "label": "Values (V):", "default": "0,10,20,30,40,50,60", "parser": lambda text: [float(v) for v in text.split(",") if v.strip()]}],
    
    "SPSA": [
        {"key": "bounds", "label": "Bounds (min,max):", "default": "0,10", "parser": lambda text: tuple(float(v) for v in text.split(",") if v.strip())},
        {"key": "a", "label": "a:", "default": "0.1", "parser": float},
        {"key": "c", "label": "c:", "default": "0.05", "parser": float},
        {"key": "A", "label": "A:", "default": "10", "parser": float},
        {"key": "alpha", "label": "alpha:", "default": "0.602", "parser": float},
        {"key": "gamma", "label": "gamma:", "default": "0.101", "parser": float},
    ],
    
    "Optuna (TPE)": [{"key": "n_startup_trials", "label": "n_startup_trials:", "default": "10", "parser": int}],
}
