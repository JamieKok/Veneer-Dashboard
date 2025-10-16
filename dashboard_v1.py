import streamlit as st
import numpy as np
import itertools
import time

# ==========================================
# PAGES
# ==========================================
page = st.sidebar.radio("📄 Pages", ["Solver", "Exploratory", "Optimiser"])

# ==========================================
# SIDEBAR INPUTS
# ==========================================
st.sidebar.title("🧮 Input Parameters")

include_265 = st.sidebar.checkbox("Include 2.65 mm veneer thickness", value=False)
mirrored_construction = st.sidebar.checkbox("Use mirrored layup", value=False)

min_thickness = st.sidebar.number_input("Min veneer thickness (mm)", 1.0, 10.0, 2.0, 0.05)
max_thickness = st.sidebar.number_input("Max veneer thickness (mm)", 1.0, 10.0, 4.0, 0.05)
increment = st.sidebar.number_input("Thickness increment (mm)", 0.01, 1.0, 0.05, 0.01)

compression_factor = st.sidebar.slider("Compression factor", 0.00, 0.20, 0.06, 0.01)
sanding_min = st.sidebar.slider("Sanding min (mm)", 0.0, 2.0, 0.75, 0.05)
sanding_max = st.sidebar.slider("Sanding max (mm)", 0.0, 2.0, 1.0, 0.05)
sanding_step = st.sidebar.slider("Sanding step (mm)", 0.01, 1.0, 0.05, 0.01)
tolerance = st.sidebar.slider("Tolerance (mm)", 0.00, 0.20, 0.05, 0.01)

# Product setup
products = [
    {"id": "9mm-5", "L": 5, "T": 9.0},
    {"id": "12mm-5", "L": 5, "T": 12.0},
    {"id": "15mm-5", "L": 5, "T": 15.0},
    {"id": "18mm-7", "L": 7, "T": 18.0},
    {"id": "21mm-7", "L": 7, "T": 21.0},
    {"id": "25mm-9", "L": 9, "T": 25.0},
    {"id": "30mm-9", "L": 9, "T": 30.0}
]

veneer_options = np.round(np.arange(min_thickness, max_thickness + 0.0001, increment), 2).tolist()

# If user checks the box, include 2.65 in the list
if include_265 and 2.65 not in veneer_options:
    veneer_options.insert(1, 2.65)  # insert it in a sensible spot

# Or if unchecked, make sure it's not in the list
elif not include_265 and 2.65 in veneer_options:
    veneer_options.remove(2.65)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def final_thickness(stack, compression_factor, sanding):
    layup = sum(stack)
    pressed = layup * (1 - compression_factor)
    return pressed - sanding

def feasible_with_sanding(stack, target):
    for sanding in np.arange(sanding_min, sanding_max + 0.0001, sanding_step):
        final = final_thickness(stack, compression_factor, sanding)
        if abs(final - target) <= tolerance:
            return True, sanding, final
    return False, None, None

def find_feasible_stack(L, target, veneers):
    for stack in itertools.combinations_with_replacement(veneers, L):
        feasible, sanding, final = feasible_with_sanding(stack, target)
        if feasible:
            return stack, sanding, final
    return None, None, None

def move_thinnest_to_faces(stack):
    stack = stack.copy()
    if len(set(stack)) == 1:
        return stack
    sorted_indices = sorted(range(len(stack)), key=lambda i: stack[i])
    front_idx = sorted_indices[0]
    stack[0], stack[front_idx] = stack[front_idx], stack[0]
    if len(stack) > 1:
        back_idx = sorted_indices[1]
        stack[-1], stack[back_idx] = stack[back_idx], stack[-1]
    return stack

def veneer_stats(veneers):
    arr = np.array(veneers)
    return {
        "min": np.min(arr),
        "max": np.max(arr),
        "gap": np.max(arr) - np.min(arr),
        "mean": np.mean(arr),
        "std": np.std(arr)
    }

def make_stack_mirrored(stack):
    L = len(stack)
    half_len = L // 2
    middle_len = L % 2
    # Keep the first half, mirror it
    first_half = stack[:half_len]
    mirrored_stack = first_half + ([stack[half_len]] if middle_len else []) + first_half[::-1]
    return mirrored_stack

# ==========================================
# PAGE 1: SOLVER
# ==========================================
if page == "Solver":
    st.title("🌲 Veneer Layup Solver")
    st.write("Search for all 3-veneer sets that can make every product within tolerance.")

    candidate_sets = list(itertools.combinations(veneer_options, 3))
    total_sets = len(candidate_sets)

    progress_bar = st.progress(0)
    valid_global_sets = []

    for idx, cset in enumerate(candidate_sets):
        all_feasible = True
        stacks_per_product = {}

        for p in products:
            stack, sanding, final = find_feasible_stack(p["L"], p["T"], cset)
            if stack is None:
                all_feasible = False
                break
            stacks_per_product[p["id"]] = {
                "stack": list(stack),
                "sanding": round(sanding, 2),
                "final": round(final, 2)
            }

        if all_feasible:
            valid_global_sets.append({"veneers": cset, "stacks": stacks_per_product})

        if idx % 100 == 0:
            progress_bar.progress((idx + 1) / total_sets)

    progress_bar.progress(1.0)
    st.success(f"✅ Found {len(valid_global_sets)} global 3-veneer sets (out of {total_sets})")

    # Adjust veneer stacks
    adjusted_global_sets = []
    for gset in valid_global_sets:
        veneers = gset["veneers"]
        adjusted_stacks = {}
        for pid, info in gset["stacks"].items():
            original_stack = info["stack"]
            adjusted_stack = move_thinnest_to_faces(original_stack)
            adjusted_stacks[pid] = {
                "original": original_stack,
                "adjusted": adjusted_stack,
                "sanding": info["sanding"],
                "final": info["final"]
            }
        adjusted_global_sets.append({
            "veneers": veneers,
            "stacks": adjusted_stacks
        })

    st.subheader("🧮 Global Solutions")

    # ✅ Display the mirrored solutions

    display_solutions = adjusted_global_sets.copy()

    # Step 1: Apply mirrored layup if checkbox is checked
    if mirrored_construction:
        for vg in display_solutions:
            for pid, info in vg["stacks"].items():
                info["adjusted"] = make_stack_mirrored(info["adjusted"])

    # Step 2: Filter for 2.65 mm if checkbox is checked
    if include_265:
        display_solutions = [
            s for s in display_solutions
            if any(abs(round(v, 2) - 2.65) < 1e-8 for v in s["veneers"])
        ]

    # Step 3: Display solutions
    if len(display_solutions) == 0:
        st.warning("⚠️ No solutions found for the selected options.")
    else:
        num_to_show = st.slider(
            "Number of solutions to display",
            1,
            min(len(display_solutions), 50),
            5
        )

        for i, vg in enumerate(display_solutions[:num_to_show], 1):

            recoveries = []
            for pid, info in vg["stacks"].items():
                raw_layup = sum(info["adjusted"])  # use mirrored if applied
                recovery = info["final"] / raw_layup * 100
                recoveries.append(recovery)
            avg_recovery = round(np.mean(recoveries), 2)

            with st.expander(f"Solution {i}: Veneers {vg['veneers']} | Avg Recovery: {avg_recovery}%"):
                for pid, info in vg["stacks"].items():
                    st.write(f"**{pid}**")
                    st.write(
                        f"Before: {info['original']} → After: {info['adjusted']} "
                        f"| Sanding: {info['sanding']} mm | Final: {info['final']} mm"
                    )

    # Store in session state
    st.session_state["adjusted_global_sets"] = adjusted_global_sets


    # # ✅ Optional post-filter: if checkbox checked, show only solutions that contain 2.65
    # display_solutions = adjusted_global_sets
    # if include_265:
    #     display_solutions = [
    #         s for s in adjusted_global_sets
    #         if any(abs(round(v, 2) - 2.65) < 1e-8 for v in s["veneers"])
    #     ]

    # if len(display_solutions) == 0:
    #     st.warning("⚠️ No solutions found that include 2.65 mm.")
    # else:
    #     num_to_show = st.slider(
    #         "Number of solutions to display",
    #         1,
    #         min(len(display_solutions), 50),
    #         5
    #     )

    #     for i, vg in enumerate(display_solutions[:num_to_show], 1):
    #         with st.expander(f"Solution {i}: Veneers {vg['veneers']}"):
    #             for pid, info in vg["stacks"].items():
    #                 st.write(f"**{pid}**")
    #                 st.write(
    #                     f"Before: {info['original']} → After: {info['adjusted']} "
    #                     f"| Sanding: {info['sanding']} mm | Final: {info['final']} mm"
    #                 )

    # # Store in session state
    # st.session_state["adjusted_global_sets"] = adjusted_global_sets

# ==========================================
# PAGE 2: EXPLORATORY
# ==========================================
elif page == "Exploratory":
    st.title("📊 Exploratory Veneer Analysis")
    if "adjusted_global_sets" not in st.session_state:
        st.warning("⚠️ Please run the Solver page first.")
    else:
        adjusted_global_sets = st.session_state["adjusted_global_sets"]

        # Compute stats for each solution
        exploratory_solutions = []
        for g in adjusted_global_sets:
            stats = veneer_stats(g["veneers"])
            exploratory_solutions.append({**g, **stats})

        smallest_gap_solution = min(exploratory_solutions, key=lambda x: x["gap"])
        largest_gap_solution = max(exploratory_solutions, key=lambda x: x["gap"])
        smallest_veneer_solution = min(exploratory_solutions, key=lambda x: x["min"])
        largest_veneer_solution = max(exploratory_solutions, key=lambda x: x["max"])
        lowest_std_solution = min(exploratory_solutions, key=lambda x: x["std"])

        choice = st.selectbox(
            "Choose exploratory view",
            ["Smallest Gap", "Largest Gap", "Smallest Veneer", "Largest Veneer", "Lowest Std Dev"]
        )

        mapping = {
            "Smallest Gap": smallest_gap_solution,
            "Largest Gap": largest_gap_solution,
            "Smallest Veneer": smallest_veneer_solution,
            "Largest Veneer": largest_veneer_solution,
            "Lowest Std Dev": lowest_std_solution
        }

        solution = mapping[choice]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Min", f"{solution['min']:.2f} mm")
        col2.metric("Max", f"{solution['max']:.2f} mm")
        col3.metric("Gap", f"{solution['gap']:.2f} mm")
        col4.metric("Mean", f"{solution['mean']:.2f} mm")
        col5.metric("Std", f"{solution['std']:.2f}")

        st.subheader(f"Veneer Set: {solution['veneers']}")
        for pid, info in solution["stacks"].items():
            with st.expander(f"{pid}"):
                st.write(f"Before: {info['original']} → After: {info['adjusted']}")
                st.write(f"Sanding: {info['sanding']} mm | Final: {info['final']} mm")

# ==========================================
# PAGE 3: OPTIMISER
# ==========================================
if page == "Optimiser":
    st.title("🌲 Layup to Sanding Optimiser")
    st.write("Find the layup construction that minimises recovery from layup to final product.")

    if "adjusted_global_sets" not in st.session_state:
        st.warning("⚠️ Please run the Solver page first.")
    else:
        adjusted_global_sets = st.session_state["adjusted_global_sets"]

        # Filter solutions
        filtered_solutions = adjusted_global_sets

        if mirrored_construction:
            filtered_solutions = [
                s for s in filtered_solutions
                if all(info["adjusted"] == move_thinnest_to_faces(info["adjusted"])
                       for info in s["stacks"].values())
            ]

        if include_265:
            filtered_solutions = [
                s for s in filtered_solutions
                if any(abs(round(v, 2) - 2.65) < 1e-8 for v in s["veneers"])
            ]

        if not filtered_solutions:
            st.warning("⚠️ No solutions meet the selected criteria.")
        else:
            # Compute recovery for each solution
            solution_recoveries = []
            for s in filtered_solutions:
                recoveries = []
                for pid, info in s["stacks"].items():
                    raw_layup = sum(info["adjusted"])
                    final = info["final"]
                    recovery = final / raw_layup * 100
                    recoveries.append(recovery)
                avg_recovery = np.mean(recoveries)
                solution_recoveries.append({**s, "avg_recovery": avg_recovery})

            # Pick solution with minimum recovery
            best_solution = min(solution_recoveries, key=lambda x: x["avg_recovery"])

            st.subheader("🏆 Optimal Solution")
            st.write(f"Average Recovery: {best_solution['avg_recovery']:.2f} %")
            st.write(f"Veneer Set: {best_solution['veneers']}")

            for pid, info in best_solution["stacks"].items():
                with st.expander(f"{pid}"):
                    st.write(f"Before: {info['original']} → After: {info['adjusted']}")
                    st.write(f"Sanding: {info['sanding']} mm | Final: {info['final']} mm")
                    raw_layup = sum(info["adjusted"])
                    recovery = info["final"] / raw_layup * 100
                    st.write(f"Recovery: {recovery:.2f} %")
