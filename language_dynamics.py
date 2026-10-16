#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import random
import sys
import string
import time  # Ensure the time module is imported
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp


# ========== 1) Data Loading Functions ==========

def load_csv_data(file_path: str) -> List[List[str]]:
    """
    Loads a CSV file into a 2D list of strings.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    List[List[str]]
        2D list where each sub-list is a row from the CSV.
    """
    data = []
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def load_parameters(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads parameters from a CSV file with columns:
      state, language, alpha1, alpha2, s, x0, error

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    List[Dict[str, Any]]
        Each dictionary has keys: 'state', 'language', 'alpha1', 'alpha2', 's', 'x0', 'error'.
    """
    parameters = []
    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                parameters.append({
                    'state':     row['state'],
                    'language':  row['language'],
                    'alpha1':    float(row['alpha1']),
                    'alpha2':    float(row['alpha2']),
                    's':         float(row['s']),
                    'x0':        float(row['x0']),
                    'error':     float(row['error'])
                })
            except KeyError as e:
                raise KeyError(f"Missing expected column in CSV: {e}")
            except ValueError as e:
                raise ValueError(f"Could not convert a value to float: {e}")
    return parameters


# ========== 2.5) Filtering Data Points ==========

def get_filtered_row_points(data: List[List[str]], row_index: int, exclude_rows: List[int]) -> List[Tuple[float, float]]:
    """
    Extract (t, y) pairs from a given row in a 2D data list,
    excluding points that appear in specified rows.

    Assumes the first row (data[0]) contains time labels from column 2 onward,
    and the target row (data[row_index]) has y values from column 2 onward.

    Parameters
    ----------
    data : List[List[str]]
        2D list of strings (CSV data).
    row_index : int
        The target row index (0-based) from which to extract y-values.
    exclude_rows : List[int]
        List of row indices (0-based) whose points are to be excluded.

    Returns
    -------
    List[Tuple[float, float]]
        A list of (t, y) pairs excluding those found in exclude_rows.
    """
    # Assume first two columns are not time/value data.
    x_labels = data[0][2:]
    y_values = data[row_index][2:]

    # Collect excluded points from the given rows.
    excluded_points = {
        (float(xl), float(yl))
        for ex_row in exclude_rows
        for xl, yl in zip(x_labels, data[ex_row][2:])
    }

    points = []
    for xl, yl in zip(x_labels, y_values):
        t = float(xl)
        val = float(yl)
        if (t, val) not in excluded_points:
            points.append((t, val))
    return points


def invert_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Given a list of (t, y) points, return a new list with (t, 1 - y).

    Parameters
    ----------
    points : List[Tuple[float, float]]
        List of (t, y) pairs.

    Returns
    -------
    List[Tuple[float, float]]
        Transformed list with y replaced by 1 - y.
    """
    return [(t, 1 - y) for (t, y) in points]


# ========== 3) ODE Definition ==========

def ode_system(t: float, x: List[float], alpha1: float, alpha2: float, s: float) -> List[float]:
    """
    Defines the ODE:
      dx/dt = (1 - x)*x^(alpha1)*s - x*(1 - x)^(alpha2)*(1 - s)

    Parameters
    ----------
    t : float
        Independent variable.
    x : List[float]
        State vector (using x[0]).
    alpha1 : float
        Exponent for x.
    alpha2 : float
        Exponent for (1 - x).
    s : float
        Parameter s.

    Returns
    -------
    List[float]
        The derivative [dx/dt].
    """
    x_val = x[0]
    dxdt = (1 - x_val) * (x_val ** alpha1) * s - x_val * ((1 - x_val) ** alpha2) * (1 - s)
    return [dxdt]


# ========== 4) Plotting Functions ==========

def plot_ode_with_points(row_num: int, data: List[List[str]], params: List[Dict[str, Any]],
                         exclude_rows: List[int] = []) -> None:
    """
    Plots the ODE solution x(t), its complement 1 - x(t), and data points from CSV.
    Optionally excludes specified points.

    Note: row_num is 1-based (i.e. data[row_num] is used and params[row_num - 1]).

    Parameters
    ----------
    row_num : int
        The 1-based row number to plot.
    data : List[List[str]]
        2D CSV data.
    params : List[Dict[str, Any]]
        Parameters with keys: 'alpha1', 'alpha2', 's', 'x0', etc.
    exclude_rows : List[int], optional
        List of 0-based row indices to exclude from data points.
    """
    # 1) Extract and filter data points.
    pointsX = get_filtered_row_points(data, row_index=row_num, exclude_rows=exclude_rows)
    pointsY = invert_points(pointsX)

    # 2) Extract parameters (note: converting 1-based row_num to 0-based index).
    selected = params[row_num - 1]
    alpha1_val = selected['alpha1']
    alpha2_val = selected['alpha2']
    s_val      = selected['s']
    x0_val     = selected['x0']

    # 3) Solve the ODE.
    def ode_wrapper(t, x):
        return ode_system(t, x, alpha1_val, alpha2_val, s_val)

    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 100)
    solution = solve_ivp(ode_wrapper, t_span, [x0_val], t_eval=t_eval)
    solution_years = 1970 + 10 * solution.t  # Convert time to years

    # 4) Plotting.
    plt.figure(figsize=(8, 5))
    plt.plot(solution_years, solution.y[0], label='ODE solution: x(t)', linewidth=2)
    plt.plot(solution_years, 1 - solution.y[0], label='1 - x(t)', linewidth=2)

    # Plot data points (X and inverted Y).
    X_ts = [pt[0] for pt in pointsX]
    X_vals = [pt[1] for pt in pointsX]
    X_years = [1970 + 10 * t for t in X_ts]
    plt.plot(X_years, X_vals, 'ro', label='pointsX', markersize=6)

    Y_ts = [pt[0] for pt in pointsY]
    Y_vals = [pt[1] for pt in pointsY]
    Y_years = [1970 + 10 * t for t in Y_ts]
    plt.plot(Y_years, Y_vals, 'go', label='pointsY (1 - x)', markersize=6)

    plt.ylim(0, 1)
    plt.xlabel('Year')
    plt.ylabel('Percentage of speakers')
    title_label = f"{data[row_num][1]} in {data[row_num][0]}"
    plt.title(title_label)
    plt.grid(True)
    plt.legend()
    plt.show()

    # 5) Compute and print RMSE error (without averaging).
    squared_diffs = [
        (np.interp(tx, solution.t, solution.y[0]) - valx) ** 2
        for (tx, valx) in pointsX
    ]
    error_value = np.sqrt(sum(squared_diffs))
    print(f"Error = {error_value}")


def plot_equation_from_params(row_num: int, params: List[Dict[str, Any]], val: float) -> None:
    """
    Plots f(x) over the interval [0, 1] for the ODE's right-hand side given a set of parameters.
    Also draws arrows along the x-axis indicating the sign of f(x).

    Parameters
    ----------
    row_num : int
        1-based index for the parameter set.
    params : List[Dict[str, Any]]
        Parameter list containing keys: 'alpha1', 'alpha2', 's', 'x0', etc.
    val : float
        Controls the y-limit for the plot ([-val, val]).
    """
    selected = params[row_num - 1]
    alpha1_val = selected['alpha1']
    alpha2_val = selected['alpha2']
    s_val      = selected['s']
    x0_val     = selected['x0']

    # Define f(x) = dx/dt.
    def f(x: float) -> float:
        return (1 - x) * (x ** alpha1_val) * s_val - x * ((1 - x) ** alpha2_val) * (1 - s_val)

    x_values = np.linspace(0, 1, 200)
    y_values = [f(x) for x in x_values]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, color='blue', label='f(x)')
    plt.plot([x0_val], [0], 'ro', label='x0')  # Mark initial point

    plt.xlim(0, 1)
    plt.ylim(-val, val)
    ax = plt.gca()
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Draw arrows at y=0 indicating the sign of f(x).
    arrow_positions = np.linspace(0.1, 0.9, 15)
    arrow_length = 0.02
    for x_pos in arrow_positions:
        fx = f(x_pos)
        dx = arrow_length if fx > 0 else -arrow_length
        plt.arrow(
            x_pos, 0,
            dx, 0,
            head_width=0.004,
            head_length=0.005,
            fc='green', ec='green',
            zorder=5
        )

    plt.xlabel('x', loc='left')
    plt.ylabel(r'$\dot{x}$')
    title_str = f"{selected['language']} in {selected['state']}"
    plt.title(title_str)
    plt.grid(True)
    plt.legend()
    plt.show()


# ========== 5) Objective Function for ODE Fitting ==========

def objective_function(data: List[List[str]], row_num: int, vector: Tuple[float, float, float, float],
                       exclude_rows: List[int] = []) -> float:
    """
    Computes the RMSE error between the ODE solution and CSV data for a given parameter vector.
    (RMSE computed without dividing by n.)

    Parameters
    ----------
    data : List[List[str]]
        CSV data as a 2D list.
    row_num : int
        1-based index for the data row.
    vector : Tuple[float, float, float, float]
        A tuple containing (alpha1, alpha2, s, x0_val).
    exclude_rows : List[int], optional
        List of row indices (0-based) to exclude. Default is empty.

    Returns
    -------
    float
        The computed error.
    """
    alpha1_val, alpha2_val, s_val, x0_val = vector
    pointsX = get_filtered_row_points(data, row_index=row_num, exclude_rows=exclude_rows)

    def ode_wrapper(t, x):
        return [(1 - x[0]) * (x[0] ** alpha1_val) * s_val - x[0] * ((1 - x[0]) ** alpha2_val) * (1 - s_val)]

    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 100)
    solution = solve_ivp(ode_wrapper, t_span, [x0_val], t_eval=t_eval)

    squared_diffs = [
        (np.interp(tx, solution.t, solution.y[0]) - valx) ** 2
        for (tx, valx) in pointsX
    ]
    error_value = np.sqrt(sum(squared_diffs))
    return error_value


# ========== 6) Utility Function to Extract Parameters ==========

def extract_values_by_index(params: List[Dict[str, Any]], row_num: int) -> Tuple[float, float, float, float]:
    """
    Extracts the parameter vector (alpha1, alpha2, s, x0) for a given row.
    Uses 1-based indexing.

    Parameters
    ----------
    params : List[Dict[str, Any]]
        List of parameter dictionaries.
    row_num : int
        1-based row number.

    Returns
    -------
    Tuple[float, float, float, float]
        (alpha1, alpha2, s, x0)
    """
    row = params[row_num - 1]
    return row['alpha1'], row['alpha2'], row['s'], row['x0']


# ========== 7) Local Neighborhood and Local Search ==========

def neighborhood(vector: Tuple[float, float, float, float], distance: float = 0.004,
                 decimals: int = 3) -> List[Tuple[float, float, float, float]]:
    """
    Generates neighbors around a given parameter vector within a specified Euclidean distance.

    Parameters
    ----------
    vector : Tuple[float, float, float, float]
        The current parameter vector (alpha1, alpha2, s, x0).
    distance : float, optional
        Maximum distance for neighbors. Default is 0.004.
    decimals : int, optional
        Number of decimals to round candidate values. Default is 3.

    Returns
    -------
    List[Tuple[float, float, float, float]]
        List of candidate neighbor vectors.
    """
    alpha1, alpha2, s, x0 = vector
    step_size = 10 ** -decimals
    steps = np.arange(-distance, distance + step_size, step_size)

    neighbors = []
    for dAlpha1, dAlpha2, dS, dX0 in product(steps, repeat=4):
        candidate = (
            round(alpha1 + dAlpha1, decimals),
            round(alpha2 + dAlpha2, decimals),
            round(s + dS, decimals),
            round(x0 + dX0, decimals)
        )
        # Only consider candidates within the specified distance (nonzero)
        if 0 < np.linalg.norm(np.subtract(candidate, vector)) <= distance:
            neighbors.append(candidate)
    return neighbors


def local_search(params: List[Dict[str, Any]], data: List[List[str]], row_num: int,
                 neighborhood_func, objective_func, distance: float = 0.002,
                 decimals: int = 3, exclude_rows: List[int] = []) -> Tuple[Tuple[float, float, float, float], float]:
    """
    Performs a single-step local search to minimize the objective function.

    Parameters
    ----------
    params : List[Dict[str, Any]]
        List of parameter dictionaries.
    data : List[List[str]]
        CSV data.
    row_num : int
        1-based row number for the data/parameters.
    neighborhood_func : callable
        Function to generate a neighborhood.
    objective_func : callable
        Objective function to minimize.
    distance : float, optional
        Maximum neighborhood distance. Default is 0.002.
    decimals : int, optional
        Rounding precision. Default is 3.
    exclude_rows : List[int], optional
        List of 0-based row indices to exclude.

    Returns
    -------
    Tuple[Tuple[float, float, float, float], float]
        Best neighbor vector and its associated error.
    """
    vector = extract_values_by_index(params, row_num)
    nbhd = neighborhood_func(vector, distance, decimals)

    evaluated_neighbors = [(neighbor, objective_func(data, row_num, neighbor, exclude_rows))
                           for neighbor in nbhd]

    best_error = min(evaluated_neighbors, key=lambda x: x[1])[1]
    best_candidates = [n for n, err in evaluated_neighbors if err == best_error]
    best_neighbor = random.choice(best_candidates)
    return best_neighbor, best_error


def iterated_local_search(params: List[Dict[str, Any]], data: List[List[str]], row_num: int,
                          neighborhood_func, objective_func, iterations: int = 10,
                          distance: float = 0.002, decimals: int = 3,
                          output_file: str = "local_search_results_with_error.csv",
                          exclude_rows: List[int] = []) -> Tuple[Tuple[float, float, float, float], float]:
    """
    Performs an iterated local search to minimize the objective function.
    Saves the intermediate best solutions to a CSV file.

    Parameters
    ----------
    params : List[Dict[str, Any]]
        List of parameter dictionaries.
    data : List[List[str]]
        CSV data.
    row_num : int
        1-based row number for optimization.
    neighborhood_func : callable
        Function to generate the neighborhood.
    objective_func : callable
        Objective function to minimize.
    iterations : int, optional
        Number of iterations (default: 10).
    distance : float, optional
        Neighborhood search distance (default: 0.002).
    decimals : int, optional
        Rounding decimals (default: 3).
    output_file : str, optional
        Path to CSV file for logging results.
    exclude_rows : List[int], optional
        Rows to exclude.

    Returns
    -------
    Tuple[Tuple[float, float, float, float], float]
        The best parameter vector and its error.
    """
    vector = extract_values_by_index(params, row_num)
    best_vector = vector
    best_error = objective_func(data, row_num, best_vector, exclude_rows)

    results = [{
        "alpha1": best_vector[0],
        "alpha2": best_vector[1],
        "s": best_vector[2],
        "x0": best_vector[3],
        "error": best_error
    }]

    for iteration in range(iterations):
        sys.stdout.write(f"\rIteration {iteration + 1}/{iterations}...")
        sys.stdout.flush()

        nbhd = neighborhood_func(best_vector, distance, decimals)
        evaluated_neighbors = [(neighbor, objective_func(data, row_num, neighbor, exclude_rows))
                               for neighbor in nbhd]

        current_min_error = min(evaluated_neighbors, key=lambda x: x[1])[1]
        best_candidates = [n for n, err in evaluated_neighbors if err == current_min_error]
        new_best_vector = random.choice(best_candidates)

        if current_min_error < best_error:
            best_error = current_min_error
            best_vector = new_best_vector
            results.append({
                "alpha1": best_vector[0],
                "alpha2": best_vector[1],
                "s": best_vector[2],
                "x0": best_vector[3],
                "error": best_error
            })
        else:
            print("\nNo further improvement, stopping early.")
            break

    with open(output_file, mode="w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["alpha1", "alpha2", "s", "x0", "error"])
        writer.writeheader()
        writer.writerows(results)

    return best_vector, best_error



# ========== 10) Updating Parameters ==========

def update_params(params: List[Dict[str, Any]], row_num: int,
                  best_vector: Tuple[float, float, float, float], best_error: float) -> None:
    """
    Updates the parameters in memory with the best found values.

    Parameters
    ----------
    params : List[Dict[str, Any]]
        List of parameter dictionaries.
    row_num : int
        1-based row number to update.
    best_vector : Tuple[float, float, float, float]
        New values (alpha1, alpha2, s, x0).
    best_error : float
        New error value.
    """
    idx = row_num - 1
    if 0 <= idx < len(params):
        required_keys = {'alpha1', 'alpha2', 's', 'x0', 'error'}
        if required_keys.issubset(params[idx].keys()):
            params[idx].update({
                'alpha1': best_vector[0],
                'alpha2': best_vector[1],
                's': best_vector[2],
                'x0': best_vector[3],
                'error': best_error
            })
        else:
            raise KeyError(
                f"Row {idx} in params is missing required keys: {required_keys}."
            )
    else:
        raise IndexError(f"Row number {row_num} is out of range for the params list.")


def save_params_to_csv(params: List[Dict[str, Any]], output_file: str) -> None:
    """
    Saves the parameter list to a CSV file.

    Parameters
    ----------
    params : List[Dict[str, Any]]
        List of parameter dictionaries.
    output_file : str
        Path to the output CSV file.
    """
    if not params:
        raise ValueError("The 'params' list is empty. Cannot save to CSV.")

    header = params[0].keys()
    with open(output_file, mode="w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=header)
        writer.writeheader()
        writer.writerows(params)
    print(f"Parameters have been saved to {output_file}")


# ========== 11) Working with the CSV Files ==========

def process_and_merge_csv_files(file_pattern: str, num_files: int, output_file: str) -> None:
    """
    Processes CSV files by removing rows with negative numbers and merging
    the last row of each processed file into a single CSV.

    Parameters
    ----------
    file_pattern : str
        Pattern for file names, e.g., 'local_search_results_row_{i}.csv'.
    num_files : int
        Number of files to process.
    output_file : str
        Path to save the merged CSV.
    """
    # Build the list of files based on the pattern.
    data_files = [file_pattern.format(i=i) for i in range(1, num_files + 1)]

    # Step 1: Process each file (filter out rows with any negative number).
    for file_name in data_files:
        try:
            df = pd.read_csv(file_name)
            filtered_df = df[(df >= 0).all(axis=1)]
            filtered_df.to_csv(file_name, index=False)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    # Step 2: Merge the last row from each processed file.
    last_rows = []
    for file_name in data_files:
        try:
            df = pd.read_csv(file_name)
            if not df.empty:
                last_rows.append(df.iloc[-1])
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")

    if last_rows:
        result_df = pd.DataFrame(last_rows)
        result_df.to_csv(output_file, index=False)
        print(f"Processing complete. Merged results saved to {output_file}.")
    else:
        print("No valid rows to save after filtering.")


# ========== 12) exhaustive_search ==========

def exhaustive_search(
    data: List[List[str]],
    row_num: int,
    objective_func,
    x0: float,
    num_alpha_points: int,
    num_s_points: int,
    output_file: str = "exhaustive_search_results.csv",
) -> Tuple[Tuple[float, float, float, float], float]:
    """
    Performs an exhaustive search over the entire grid defined by:
      - alpha1 from 0 to 2,
      - alpha2 from 0 to 2,
      - s from 0 to 1,
      with the constraint that alpha1 >= alpha2.
    The fourth parameter, x0, is kept constant.

    For every candidate (alpha1, alpha2, s, x0), the objective function is evaluated.
    The function logs each candidate and its associated error in an output CSV file.
    Progress is shown by printing the number of remaining grid points after each evaluation.

    Parameters
    ----------
    data : List[List[str]]
        The CSV data as a 2D list of strings.
    row_num : int
        The 1-based row number to be used with the objective function.
    objective_func : callable
        A function that accepts (data, row_num, candidate, exclude_rows) and returns an error value.
    x0 : float
        The constant value to be used as the fourth parameter in every candidate.
    num_alpha_points : int
        The number of grid points to use for both alpha1 and alpha2 over [0, 2].
    num_s_points : int
        The number of grid points to use for s over [0, 1].
    output_file : str, optional
        Path to the CSV file where results will be saved (default is "exhaustive_search_results.csv").

    Returns
    -------
    Tuple[Tuple[float, float, float, float], float]
        A tuple containing the best candidate vector (alpha1, alpha2, s, x0) and its associated error.
    """
    # Create grid values.
    alpha1_values = np.linspace(0, 2, num_alpha_points)
    alpha2_values = np.linspace(0, 2, num_alpha_points)
    s_values      = np.linspace(0, 1, num_s_points)

    # Count total number of valid candidates (alpha1 >= alpha2).
    total_candidates = 0
    for a1 in alpha1_values:
        for a2 in alpha2_values:
            if a1 >= a2:
                total_candidates += len(s_values)

    remaining = total_candidates

    best_vector = None
    best_error = float('inf')
    results = []  # To store dictionaries for each evaluated candidate

    # Loop over the grid.
    for a1 in alpha1_values:
        for a2 in alpha2_values:
            if a1 >= a2:  # Respect the constraint.
                for s in s_values:
                    candidate = (float(a1), float(a2), float(s), x0)
                    error_val = objective_func(data, row_num, candidate)

                    # Log the candidate and its error.
                    results.append({
                        "alpha1": candidate[0],
                        "alpha2": candidate[1],
                        "s": candidate[2],
                        "x0": candidate[3],
                        "error": error_val
                    })

                    # Update best candidate if necessary.
                    if error_val < best_error:
                        best_error = error_val
                        best_vector = candidate

                    # Update progress.
                    remaining -= 1
                    sys.stdout.write(f"\rRemaining points to evaluate: {remaining}")
                    sys.stdout.flush()
    print()  # Move to a new line after progress is done.

    # Save all candidate results to a CSV file.
    with open(output_file, mode="w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["alpha1", "alpha2", "s", "x0", "error"])
        writer.writeheader()
        writer.writerows(results)

    return best_vector, best_error


# Example usage:
# data = load_csv_data("data/census_data.csv")
# params = load_parameters("data/parameters.csv")
# exhaustive_search(data, 1, objective_func=objective_function, x0=params[0]['x0'],
#                   num_alpha_points=40, num_s_points=40)
# plot_ode_with_points(1, data, params)
