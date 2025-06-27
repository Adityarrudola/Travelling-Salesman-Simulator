# 🌍 Traveling Salesman Problem (TSP) Solver – Streamlit App

A powerful and interactive web-based application to visualize and solve the **Traveling Salesman Problem (TSP)** using multiple algorithms:

🔹 Brute Force  
🔹 Greedy  
🔹 Backtracking  
🔹 Branch & Bound  
🔹 Dynamic Programming

---

## 🔗 Live Demo  
🌐 [Launch App](https://travelling-salesman-simulator-uhrwejo2pbhmu3cvedjotj.streamlit.app/)

---

## 📸 Preview  
![Preview Image](https://github.com/user-attachments/assets/704e3253-e66f-4620-8259-1b985d5e23fa)

---

## 🛠️ Built With

- Python 3  
- [Streamlit](https://streamlit.io/)  
- NumPy  
- Pandas  
- Matplotlib  
- `PriorityQueue` (from Python’s `queue` module)

---

## 📂 Features

✅ Add cities manually or randomly  
✅ Choose algorithm (Brute Force, Greedy, etc.)  
✅ Visualize shortest and longest routes  
✅ Interactive Matplotlib plots  
✅ Export routes as downloadable CSV  
✅ Algorithm complexity explainer via expanders  

---

## 💡 What I Learned

- Comparative implementation of TSP algorithms  
- Real-time optimization visualization  
- Building modular and interactive Streamlit apps  
- Exporting runtime data dynamically  
- Enhancing user experience with UI/UX improvements  

---

## 📁 Folder Structure

```
tsp-streamlit-app/
├── tsp_app.py           # Main Streamlit app
├── requirements.txt     # Dependency list
└── README.md            # Project documentation
```

---

## 🚀 Getting Started

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/tsp-streamlit-app.git
   cd tsp-streamlit-app
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**  
   ```bash
   streamlit run tsp_app.py
   ```

4. **Open in browser:**  
   `http://localhost:8501`

---

## 🧠 Algorithm Comparison

| Algorithm           | Time Complexity      | Guarantees Optimal? |
|---------------------|----------------------|----------------------|
| Brute Force         | O(n!)                | ✅ Yes               |
| Greedy (Nearest)    | O(n²)                | ❌ No                |
| Backtracking        | O(n!)                | ✅ Yes               |
| Branch & Bound      | O(n!) (with pruning) | ✅ Yes               |
| Dynamic Programming | O(n²·2ⁿ)             | ✅ Yes               |
