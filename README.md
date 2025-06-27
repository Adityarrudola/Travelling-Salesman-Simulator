# 🌍 Traveling Salesman Problem (TSP) Solver - Streamlit App

A powerful and interactive web-based application to visualize and solve the **Traveling Salesman Problem (TSP)** using various algorithms, including brute force, greedy, backtracking, branch & bound, and dynamic programming.

---

## 🔗 Live Demo  
🌐 

---

## 📸 Preview  


---

## 🛠️ Built With

- Python 3  
- [Streamlit](https://streamlit.io/)  
- NumPy  
- Pandas  
- Matplotlib  
- PriorityQueue (from `queue` module)

---

## 📂 Features

- ✅ Add cities manually or generate randomly  
- ✅ Choose solving method (Brute Force, Greedy, Backtracking, Branch and Bound, Dynamic Programming)  
- ✅ Visualize the best and worst paths  
- ✅ Export results as CSV  
- ✅ Expandable explanations of each algorithm  
- ✅ Interactive plots using Matplotlib  

---

## 💡 What I Learned

- Implementing and comparing TSP algorithms  
- Real-time visualization of optimization problems  
- Building interactive web apps with Streamlit  
- Exporting dynamic data as downloadable files  
- Improving code modularity and user experience  

---

## 📁 Folder Structure

```
project/
├── tsp_app.py               # Main Streamlit app
├── requirements.txt         # Required libraries
├── README.md
```

---

## 🚀 Getting Started

1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/tsp-streamlit-app.git
   cd tsp-streamlit-app
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app  
   ```bash
   streamlit run tsp_app.py
   ```

4. Open in browser at `http://localhost:8501`

---

## 🧠 Algorithm Time Complexities

| Algorithm           | Time Complexity        | Guarantees Optimal? |
|---------------------|------------------------|----------------------|
| Brute Force         | O(n!)                  | ✅ Yes               |
| Greedy (Nearest)    | O(n²)                  | ❌ No                |
| Backtracking        | O(n!)                  | ✅ Yes               |
| Branch and Bound    | O(n!) (with pruning)   | ✅ Yes               |
| Dynamic Programming | O(n²·2ⁿ)               | ✅ Yes               |


