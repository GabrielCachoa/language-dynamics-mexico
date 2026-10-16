# Modeling Language Dynamics in Mexico: An Abrams–Strogatz Model with Independent Volatilities

Code, data and manuscript for the PhD thesis of **Gabriel Cachoa Ocampo**
(Programa de Maestría y Doctorado en Ciencias Matemáticas, UNAM, 2026),
directed by Dra. Bibiana Obregón Quintana.

The thesis studies an extension of the Abrams–Strogatz language competition model
in which the single volatility exponent is replaced by two independent exponents,
one for each direction of language shift:

```
dx/dt = s · x^α1 · (1 − x) − (1 − s) · (1 − x)^α2 · x
```

It gives the complete analytic classification of the (α1, α2) parameter plane and
calibrates the model with INEGI census data (1930–2020) for fifteen
language-state pairs (seven indigenous languages, twelve Mexican states).

## Repository contents

| Path | Description |
|------|-------------|
| `thesis/Gabriel_Cachoa_PhD_Thesis.pdf` | Full text of the thesis |
| `src/language_dynamics.py` | Model definition, objective function, grid search, local search and plotting |
| `data/` | Census-derived input series and fitted parameters (see format below) |
| `requirements.txt` | Python dependencies |

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Python 3.9 or later is recommended.

## Usage

```python
from src.language_dynamics import *

data   = load_csv_data("data/census_data.csv")
params = load_parameters("data/parameters.csv")

# Exhaustive grid search over (alpha1, alpha2, s) for the first language-state pair
best, err = exhaustive_search(data, 1, objective_func=objective_function,
                              x0=params[0]["x0"], num_alpha_points=41, num_s_points=21)

# Local refinement around the current parameter values
best, err = iterated_local_search(params, data, 1, neighborhood, objective_function,
                                  iterations=10, distance=0.002)
update_params(params, 1, best, err)

# Plots: fitted trajectory vs. data, and phase portrait
plot_ode_with_points(1, data, params)
plot_equation_from_params(1, params, val=0.3)
```

### Data format

`census_data.csv`: the first row holds the time labels from the third column onward
(t = 0 corresponds to 1970 and one unit to ten years); each following row is one
language-state pair, with the state in column 1, the language in column 2 and the
speaker fractions `x(t)` (indigenous monolinguals + bilinguals, divided by total
population) in the remaining columns.

`parameters.csv`: columns `state, language, alpha1, alpha2, s, x0, error`, one row
per language-state pair, in the same order as `census_data.csv`.

## Related publication

G. Cachoa Ocampo and B. Obregón Quintana, *An Abrams–Strogatz model with independent
volatilities*, The European Physical Journal B (2026).

## Citing

If you use this code or the thesis, please cite:

```bibtex
@phdthesis{cachoa2026language,
  author = {Cachoa Ocampo, Gabriel},
  title  = {Modeling Language Dynamics in Mexico: An Abrams--Strogatz Model with Independent Volatilities},
  school = {Universidad Nacional Aut{\'o}noma de M{\'e}xico},
  year   = {2026},
  type   = {PhD thesis}
}
```

## Funding

This work was supported by DGAPA-UNAM-PAPIIT grant IN114724 and by a SECIHTI
doctoral fellowship (No. 997614).

## License

Code: MIT License. Thesis text and figures: CC BY 4.0. Census data: INEGI (public domain,
see [inegi.org.mx](https://www.inegi.org.mx)).
