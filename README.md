# 🧠 Python + ML + Snakemake Bootcamp

Welcome to your personalized **8-week learning roadmap** to gain fluency in Python, get hands-on experience with machine learning, and learn Snakemake and GitHub tracking for reproducible pipelines.

---

## 📁 Repository Structure
```
python-ml-snakemake-bootcamp/
├── week01-02_python_basics/
├── week03-04_data_pandas_git/
├── week05-06_machine_learning/
├── week07_snakemake_pipeline/
├── week08_deep_learning_optional/
├── datasets/
├── README.md
└── environment.yml
```

---

## 📅 Weekly Breakdown

### ✅ Week 1–2: Python Basics
**Folder:** `week01-02_python_basics`

- Python syntax, loops, functions, OOP
- Data types, file I/O, plotting
- Practice scripts to clean and plot CSV data

> 🔗 Resource: [Automate the Boring Stuff](https://automatetheboringstuff.com/)
> 🎥 Video: [Python Full Course for Beginners (freeCodeCamp)](https://www.youtube.com/watch?v=rfscVS0vtbw)

### ✅ Week 3–4: Data Analysis + GitHub
**Folder:** `week03-04_data_pandas_git`

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- Git: init, commit, push, branching

> 🔗 Resource: [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
> 🎥 Video: [Data Analysis with Pandas (freeCodeCamp)](https://www.youtube.com/watch?v=vmEHCJofslg)
> 🎥 Video: [Git & GitHub Crash Course (Traversy Media)](https://www.youtube.com/watch?v=SWYqp7iY_Tc)

### ✅ Week 5–6: Machine Learning with scikit-learn
**Folder:** `week05-06_machine_learning`

- Regression, classification, train/test split
- Metrics: accuracy, precision, recall
- Practice with datasets from UCI or Kaggle

> 🔗 Resource: [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
> 🔗 Resource: [scikit-learn tutorials](https://scikit-learn.org/stable/tutorial/index.html)
> 🎥 Video: [Machine Learning Full Course - 6 Hours (freeCodeCamp)](https://www.youtube.com/watch?v=Gv9_4yMHFhI)

### ✅ Week 7: Snakemake & Pipelines
**Folder:** `week07_snakemake_pipeline`

- Write your first Snakefile:
```python
# Snakefile
rule all:
    input:
        "results/plot.png"

rule preprocess:
    input:
        "data/raw.csv"
    output:
        "data/clean.csv"
    shell:
        "python scripts/clean.py {input} {output}"

rule plot:
    input:
        "data/clean.csv"
    output:
        "results/plot.png"
    shell:
        "python scripts/plot.py {input} {output}"
```
- Create a `config.yaml` and pass parameters to your rules
- Track versions using GitHub

> 🔗 Resource: [Snakemake Tutorial](https://snakemake.readthedocs.io/en/stable/tutorial/tutorial.html)
> 🎥 Video: [Snakemake Workflow Engine Crash Course](https://www.youtube.com/watch?v=2u0r2fA9Xh4)

### ✅ Week 8: (Optional) Deep Learning with PyTorch
**Folder:** `week08_deep_learning_optional`

- PyTorch tensors, layers, and training
- Build a digit classifier with MNIST

> 🔗 Resource: [PyTorch 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
> 🎥 Video: [Deep Learning with PyTorch (freeCodeCamp)](https://www.youtube.com/watch?v=GIsg-ZUy0MY)

---

## 🧪 Example Pipeline Project (Week 7)

### Goal: Analyze a methylation CSV and output a clean plot

**Structure:**
```
snakemake-pipeline/
├── data/
│   └── raw.csv
├── scripts/
│   ├── clean.py
│   └── plot.py
├── results/
├── config.yaml
└── Snakefile
```

**Snakefile Example:** (see above)

**clean.py**:
```python
import pandas as pd
import sys
raw = pd.read_csv(sys.argv[1])
clean = raw.dropna()
clean.to_csv(sys.argv[2], index=False)
```

**plot.py**:
```python
import pandas as pd
import matplotlib.pyplot as plt
import sys

clean = pd.read_csv(sys.argv[1])
plt.plot(clean['x'], clean['y'])
plt.savefig(sys.argv[2])
```

To run:
```bash
snakemake --cores 1
```

---

## 🧩 Set Up Your Environment
```bash
conda create -n pyml python=3.10
conda activate pyml
conda install numpy pandas matplotlib seaborn scikit-learn snakemake jupyterlab git
```

Or use this `environment.yml`:
```yaml
name: pyml
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - snakemake
  - jupyterlab
  - git
```

---

## ✅ Tips for Publishing Your Pipeline
- Use a clean `README.md` to explain:
  - Input files
  - Output structure
  - How to run (`snakemake`, `conda`, etc.)
- Add `.gitignore` to avoid tracking data files
- Use GitHub Actions (later) for automatic runs/test checks
- Add `LICENSE` and `requirements.txt` or `environment.yml`

---

Let me know if you want this exported as a starter repo with example files! You can start by copying this to your GitHub `README.md`.
