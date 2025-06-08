import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import time
import pandas as pd
import random
import copy
import io
import base64

# Function to calculate Euclidean distance
def calculate_distance(city1, city2):
    """
    Calculate the Euclidean distance between two cities.
    """
    return np.linalg.norm(np.array(city1) - np.array(city2))

# Function to solve the TSP using brute force
def solve_tsp_brute_force(cities):
    """
    Solve the TSP using the brute-force approach.
    Returns the shortest path, its distance, the longest path, and its distance.
    """
    shortest_path = None
    longest_path = None
    min_distance = float('inf')
    max_distance = 0

    # Generate all possible permutations of cities
    for path in permutations(cities):
        distance = 0
        for i in range(len(path) - 1):
            distance += calculate_distance(path[i], path[i + 1])
        distance += calculate_distance(path[-1], path[0])  # Return to the starting city

        # Update the shortest and longest paths
        if distance < min_distance:
            min_distance = distance
            shortest_path = path
        if distance > max_distance:
            max_distance = distance
            longest_path = path

    return shortest_path, min_distance, longest_path, max_distance

# Function to solve the TSP using greedy algorithm
def solve_tsp_greedy(cities):
    """
    Solve the TSP using a greedy algorithm (nearest neighbor).
    Returns the shortest path, its distance, the longest path, and its distance.
    """
    if len(cities) < 2:
        return cities, 0, cities, 0

    # For shortest path
    unvisited = cities.copy()
    current_city = unvisited.pop(0)
    shortest_path = [current_city]
    min_distance = 0

    while unvisited:
        nearest_city = min(unvisited, key=lambda city: calculate_distance(current_city, city))
        min_distance += calculate_distance(current_city, nearest_city)
        current_city = nearest_city
        shortest_path.append(current_city)
        unvisited.remove(current_city)

    min_distance += calculate_distance(shortest_path[-1], shortest_path[0])

    # For longest path (reverse greedy)
    unvisited = cities.copy()
    current_city = unvisited.pop(0)
    longest_path = [current_city]
    max_distance = 0

    while unvisited:
        farthest_city = max(unvisited, key=lambda city: calculate_distance(current_city, city))
        max_distance += calculate_distance(current_city, farthest_city)
        current_city = farthest_city
        longest_path.append(current_city)
        unvisited.remove(current_city)

    max_distance += calculate_distance(longest_path[-1], longest_path[0])

    return shortest_path, min_distance, longest_path, max_distance

# Function to solve the TSP using backtracking
def solve_tsp_backtracking(cities):
    """
    Solve the TSP using backtracking.
    Returns the shortest path, its distance, the longest path, and its distance.
    """
    n = len(cities)
    best_path = None
    best_distance = float('inf')
    worst_path = None
    worst_distance = 0

    def backtrack(path, current_distance, remaining):
        nonlocal best_path, best_distance, worst_path, worst_distance
        
        if not remaining:
            final_distance = current_distance + calculate_distance(path[-1], path[0])
            if final_distance < best_distance:
                best_distance = final_distance
                best_path = path.copy()
            if final_distance > worst_distance:
                worst_distance = final_distance
                worst_path = path.copy()
            return

        for city in remaining:
            new_path = path.copy()
            new_path.append(city)
            new_remaining = remaining.copy()
            new_remaining.remove(city)
            new_distance = current_distance
            if len(path) > 0:
                new_distance += calculate_distance(path[-1], city)
            backtrack(new_path, new_distance, new_remaining)

    backtrack([], 0, cities.copy())
    return best_path, best_distance, worst_path, worst_distance

# Function to solve the TSP using branch and bound
def solve_tsp_branch_and_bound(cities):
    """
    Solve the TSP using branch and bound.
    Returns the shortest path, its distance, the longest path, and its distance.
    """
    n = len(cities)
    if n < 2:
        return cities, 0, cities, 0

    # For shortest path
    best_path = None
    best_distance = float('inf')

    # Priority queue: (lower_bound, path, distance, remaining)
    from queue import PriorityQueue
    pq = PriorityQueue()

    # Initial lower bound (minimum spanning tree)
    def calculate_initial_bound(remaining):
        if len(remaining) < 2:
            return 0
        distances = []
        for i in range(len(remaining)):
            for j in range(i+1, len(remaining)):
                distances.append(calculate_distance(remaining[i], remaining[j]))
        distances.sort()
        return sum(distances[:len(remaining)-1])

    initial_bound = calculate_initial_bound(cities)
    pq.put((initial_bound, [], 0, cities.copy()))

    while not pq.empty():
        bound, path, current_distance, remaining = pq.get()

        if bound >= best_distance:
            continue

        if not remaining:
            final_distance = current_distance + calculate_distance(path[-1], path[0])
            if final_distance < best_distance:
                best_distance = final_distance
                best_path = path.copy()
            continue

        for city in remaining:
            new_path = path.copy()
            new_path.append(city)
            new_remaining = remaining.copy()
            new_remaining.remove(city)
            new_distance = current_distance
            if len(path) > 0:
                new_distance += calculate_distance(path[-1], city)
            new_bound = new_distance + calculate_initial_bound(new_remaining)
            if new_bound < best_distance:
                pq.put((new_bound, new_path, new_distance, new_remaining))

    # For longest path (using same approach but maximizing)
    worst_path = None
    worst_distance = 0

    def calculate_max_bound(remaining):
        if len(remaining) < 2:
            return 0
        distances = []
        for i in range(len(remaining)):
            for j in range(i+1, len(remaining)):
                distances.append(calculate_distance(remaining[i], remaining[j]))
        distances.sort(reverse=True)
        return sum(distances[:len(remaining)-1])

    pq = PriorityQueue()
    initial_bound = calculate_max_bound(cities)
    pq.put((-initial_bound, [], 0, cities.copy()))  # Using negative for max priority

    while not pq.empty():
        neg_bound, path, current_distance, remaining = pq.get()
        bound = -neg_bound

        if bound <= worst_distance:
            continue

        if not remaining:
            final_distance = current_distance + calculate_distance(path[-1], path[0])
            if final_distance > worst_distance:
                worst_distance = final_distance
                worst_path = path.copy()
            continue

        for city in remaining:
            new_path = path.copy()
            new_path.append(city)
            new_remaining = remaining.copy()
            new_remaining.remove(city)
            new_distance = current_distance
            if len(path) > 0:
                new_distance += calculate_distance(path[-1], city)
            new_bound = new_distance + calculate_max_bound(new_remaining)
            if new_bound > worst_distance:
                pq.put((-new_bound, new_path, new_distance, new_remaining))

    return best_path, best_distance, worst_path, worst_distance

# Function to solve the TSP using dynamic programming
def solve_tsp_dynamic_programming(cities):
    """
    Solve the TSP using the dynamic programming approach.
    Returns the shortest path, its distance, the longest path, and its distance.
    """
    n = len(cities)
    all_points_set = set(range(n))

    # Memoization table
    memo = {}

    def dp(start, points, is_max=False):
        if (start, points, is_max) in memo:
            return memo[(start, points, is_max)]

        if not points:
            return calculate_distance(cities[start], cities[0]), [start, 0]

        if is_max:
            best_dist = 0
        else:
            best_dist = float('inf')
        best_path = []

        for next_point in points:
            remaining_points = tuple(point for point in points if point != next_point)
            dist, path = dp(next_point, remaining_points, is_max)
            total_dist = calculate_distance(cities[start], cities[next_point]) + dist

            if is_max:
                if total_dist > best_dist:
                    best_dist = total_dist
                    best_path = [start] + path
            else:
                if total_dist < best_dist:
                    best_dist = total_dist
                    best_path = [start] + path

        memo[(start, points, is_max)] = (best_dist, best_path)
        return best_dist, best_path

    # Start from the first city (index 0)
    initial_points = tuple(all_points_set - {0})

    # Calculate shortest path
    min_distance, min_path = dp(0, initial_points, is_max=False)
    shortest_path = [cities[i] for i in min_path]

    # Calculate longest path
    max_distance, max_path = dp(0, initial_points, is_max=True)
    longest_path = [cities[i] for i in max_path]

    return shortest_path, min_distance, longest_path, max_distance

# Function to plot a route on a 2D map
def plot_route_2d(ax, route, color, label, title):
    """
    Plot a route on a 2D map.
    """
    x = [city[0] for city in route]
    y = [city[1] for city in route]
    x.append(route[0][0])  # Return to the starting city
    y.append(route[0][1])
    ax.plot(x, y, color=color, linestyle="-", marker="o", markersize=8, linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")

# Function to export routes to a CSV file
def export_routes_to_csv(shortest_path, longest_path):
    """
    Export the shortest and longest routes to a CSV file.
    Returns a DataFrame that can be used for download.
    """
    data = {
        "City": [i + 1 for i in range(len(shortest_path))],
        "Shortest Path X": [city[0] for city in shortest_path],
        "Shortest Path Y": [city[1] for city in shortest_path],
        "Longest Path X": [city[0] for city in longest_path] if longest_path else [None]*len(shortest_path),
        "Longest Path Y": [city[1] for city in longest_path] if longest_path else [None]*len(shortest_path),
    }
    return pd.DataFrame(data)

# Main function
def main():
    st.set_page_config(page_title="TSP Solver", page_icon="🌍", layout="wide")
    
    # About section    
    st.title("🌍 Traveling Salesman Problem Solver")
    st.write("""
    The **Traveling Salesman Problem (TSP)** is one of the most famous problems in computer science and operations research. It seeks to find the shortest possible route that visits a set of cities exactly once and returns to the origin city. The TSP has numerous real-world applications, including logistics, route planning, and network optimization.
    """)
    
    # Expandable sections for each approach
    with st.expander("### Brute Force Approach"):
        st.write("""
        #### How It Works:
        The Brute Force Approach evaluates all possible permutations of cities to find the shortest route. It guarantees an optimal solution but is computationally expensive.

        #### Time Complexity: **O(n!)**
        - Evaluates all possible routes.
        - Suitable for small datasets (e.g., less than 10 cities).

        #### Space Complexity: **O(n)**
        - Stores the current permutation of cities.

        #### Advantages:
        - Guarantees the optimal solution.

        #### Disadvantages:
        - Computationally expensive for large datasets.
        """)

    with st.expander("### Greedy Algorithm"):
        st.write("""
        #### How It Works:
        The Greedy Algorithm (Nearest Neighbor) builds a route by always choosing the nearest unvisited city next. It's fast but doesn't guarantee an optimal solution.

        #### Time Complexity: **O(n^2)**
        - Much faster than brute force.
        - Suitable for medium to large datasets.

        #### Space Complexity: **O(n)**
        - Stores the current route and unvisited cities.

        #### Advantages:
        - Very fast execution.
        - Good for quick approximations.

        #### Disadvantages:
        - Doesn't guarantee optimal solution.
        - Can get stuck in local optima.
        """)

    with st.expander("### Backtracking Approach"):
        st.write("""
        #### How It Works:
        Backtracking builds routes incrementally and abandons a route ("backtracks") when it determines the route cannot possibly be optimal.

        #### Time Complexity: **O(n!)**
        - Same as brute force in worst case.
        - Can be much better with good pruning.

        #### Space Complexity: **O(n)**
        - Stores the current path being explored.

        #### Advantages:
        - Can be more efficient than brute force with pruning.
        - Guarantees optimal solution.

        #### Disadvantages:
        - Still computationally expensive for large datasets.
        """)

    with st.expander("### Branch and Bound Approach"):
        st.write("""
        #### How It Works:
        Branch and Bound systematically explores routes while using bounds to eliminate suboptimal paths early.

        #### Time Complexity: **O(n!)**
        - Same as brute force in worst case.
        - Typically much better with good bounding.

        #### Space Complexity: **O(n)**
        - Stores the priority queue of paths to explore.

        #### Advantages:
        - More efficient than brute force with good bounds.
        - Guarantees optimal solution.

        #### Disadvantages:
        - Still computationally expensive for very large datasets.
        """)

    with st.expander("### Dynamic Programming Approach"):
        st.write("""
        #### How It Works:
        Dynamic Programming (Held-Karp algorithm) uses memoization to store intermediate results and avoid redundant calculations.

        #### Time Complexity: **O(n^2 * 2^n)**
        - More efficient than brute force for medium-sized datasets.

        #### Space Complexity: **O(n * 2^n)**
        - Stores intermediate results in a memoization table.

        #### Advantages:
        - More efficient than brute force for medium-sized datasets.
        - Guarantees optimal solution.

        #### Disadvantages:
        - Still computationally expensive for very large datasets.
        """)

    st.write("---")
    st.write("Add cities manually or generate them randomly. Then click 'Solve TSP' to find the shortest and longest paths.")
    
    # Initialize session state to store cities
    if "cities" not in st.session_state:
        st.session_state.cities = []

    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        if st.button("Add Random City"):
            x, y = np.random.randint(0, 100, 2)
            st.session_state.cities.append((x, y))

        if st.button("Clear Cities"):
            st.session_state.cities = []

        st.write("### Add City Manually")
        x = st.slider("X Coordinate", 0, 100, 50)
        y = st.slider("Y Coordinate", 0, 100, 50)
        if st.button("Add City at (X, Y)"):
            st.session_state.cities.append((x, y))

        # Dropdown to select solving approach
        approach = st.selectbox("Select Solving Approach", 
                               ["Brute Force", "Greedy Algorithm", "Backtracking", "Branch and Bound", "Dynamic Programming"])

    # Display the map for adding cities
    st.write("### Map")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Cities and Routes")
    ax.set_aspect("equal")

    # Plot cities
    if st.session_state.cities:
        x = [city[0] for city in st.session_state.cities]
        y = [city[1] for city in st.session_state.cities]
        ax.scatter(x, y, color="blue", s=100)
        for i, city in enumerate(st.session_state.cities):
            ax.text(city[0] + 1, city[1] + 1, f"{i + 1}", fontsize=12, color="red")

    # Display the map for adding cities
    st.pyplot(fig)

    # Solve TSP and plot the best and worst routes
    if st.button("Solve TSP"):
        if len(st.session_state.cities) < 2:
            st.warning("Please add at least 2 cities.")
        else:
            start_time = time.time()

            if approach == "Brute Force":
                shortest_path, min_distance, longest_path, max_distance = solve_tsp_brute_force(st.session_state.cities)
            elif approach == "Greedy Algorithm":
                shortest_path, min_distance, longest_path, max_distance = solve_tsp_greedy(st.session_state.cities)
            elif approach == "Backtracking":
                shortest_path, min_distance, longest_path, max_distance = solve_tsp_backtracking(st.session_state.cities)
            elif approach == "Branch and Bound":
                shortest_path, min_distance, longest_path, max_distance = solve_tsp_branch_and_bound(st.session_state.cities)
            elif approach == "Dynamic Programming":
                shortest_path, min_distance, longest_path, max_distance = solve_tsp_dynamic_programming(st.session_state.cities)

            end_time = time.time()
            st.success(f"Shortest Path Distance: {min_distance:.2f}")
            if longest_path:
                st.warning(f"Longest Path Distance: {max_distance:.2f}")
            st.info(f"Time taken: {end_time - start_time:.2f} seconds")

            # Create two columns for side-by-side plots
            col1, col2 = st.columns(2)

            # Plot best route
            with col1:
                st.write("### Best Route (Shortest Path)")
                fig_best, ax_best = plt.subplots(figsize=(6, 6))
                plot_route_2d(ax_best, shortest_path, color="green", label="Shortest Path", title="Best Route (Shortest Path)")
                ax_best.scatter([city[0] for city in st.session_state.cities], [city[1] for city in st.session_state.cities], color="blue", s=100)
                for i, city in enumerate(st.session_state.cities):
                    ax_best.text(city[0] + 1, city[1] + 1, f"{i + 1}", fontsize=12, color="red")
                ax_best.legend()
                st.pyplot(fig_best)

            # Plot worst route
            with col2:
                if longest_path:
                    st.write("### Worst Route (Longest Path)")
                    fig_worst, ax_worst = plt.subplots(figsize=(6, 6))
                    plot_route_2d(ax_worst, longest_path, color="red", label="Longest Path", title="Worst Route (Longest Path)")
                    ax_worst.scatter([city[0] for city in st.session_state.cities], [city[1] for city in st.session_state.cities], color="blue", s=100)
                    for i, city in enumerate(st.session_state.cities):
                        ax_worst.text(city[0] + 1, city[1] + 1, f"{i + 1}", fontsize=12, color="red")
                    ax_worst.legend()
                    st.pyplot(fig_worst)

            # Export routes to CSV
            csv_data = export_routes_to_csv(shortest_path, longest_path if longest_path else shortest_path)
            csv = csv_data.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Routes as CSV",
                data=csv,
                file_name="tsp_routes.csv",
                mime="text/csv",
            )

    # Display the list of cities in a table
    st.write("### List of Cities")
    if st.session_state.cities:
        city_data = {
            "City Name": [f"City {i + 1}" for i in range(len(st.session_state.cities))],
            "X Axis": [city[0] for city in st.session_state.cities],
            "Y Axis": [city[1] for city in st.session_state.cities],
        }
        df = pd.DataFrame(city_data)
        st.table(df)
    else:
        st.write("No cities added yet.")

if __name__ == "__main__":
    main()