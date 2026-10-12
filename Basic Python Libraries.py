import math
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("========== MATH LIBRARY ==========")
number = 25

print(f"Square root of {number} : {math.sqrt(number)}")
print(f"Power (2^5)       : {math.pow(2, 5)}")
print(f"Factorial of 5    : {math.factorial(5)}")
print(f"Value of PI       : {math.pi:.4f}")
print(f"Ceiling of 4.3    : {math.ceil(4.3)}")
print(f"Floor of 4.7      : {math.floor(4.7)}")

print("\n========== NUMPY LIBRARY ==========")
marks = np.array([65, 72, 80, 75, 90])

print(f"Marks             : {marks}")
print(f"Sum               : {np.sum(marks)}")
print(f"Mean              : {np.mean(marks):.2f}")
print(f"Median            : {np.median(marks):.2f}")
print(f"Std Deviation     : {np.std(marks):.2f}")
print(f"Maximum           : {np.max(marks)}")
print(f"Minimum           : {np.min(marks)}")

print("\n========== SCIPY LIBRARY ==========")
z_scores = stats.zscore(marks)
mode_result = stats.mode(marks, keepdims=True)

print(f"Z-Scores          : {np.round(z_scores, 2)}")
print(f"Mode              : {mode_result.mode[0]}")

print("\n========== PANDAS LIBRARY ==========")
data = {
    "Student": ["Amit", "Neha", "Rahul", "Sneha", "Priya"],
    "Maths": [78, 85, 72, 90, 88],
    "Python": [82, 90, 75, 95, 86],
    "AI": [80, 88, 70, 92, 90],
}

df = pd.DataFrame(data)
df["Average"] = df[["Maths", "Python", "AI"]].mean(axis=1)

print("\nStudent Data with Average:")
print(df.to_string(index=False))

print("\nSummary Statistics:")
print(df.describe().round(2))

print("\n========== MATPLOTLIB ==========")
plt.figure(figsize=(7, 4))
plt.bar(df["Student"], df["Average"], color="steelblue", edgecolor="black")
plt.xlabel("Students")
plt.ylabel("Average Marks")
plt.title("Average Marks of Students")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

print("\n========== SEABORN ==========")
sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="Maths", y="Python", hue="Student", s=120)
plt.title("Maths Marks vs Python Marks")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
correlation = df[["Maths", "Python", "AI"]].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Between Subjects")
plt.tight_layout()
plt.show()



#========== MATH LIBRARY ==========
#Square root of 25 : 5.0
#Power (2^5)       : 32.0
#Factorial of 5    : 120
#Value of PI       : 3.1416
#Ceiling of 4.3    : 5
#Floor of 4.7      : 4

#========== NUMPY LIBRARY ==========
#Marks             : [65 72 80 75 90]
#Sum               : 382
#Mean              : 76.40
#Median            : 75.00
#Std Deviation     : 8.36
#Maximum           : 90
#Minimum           : 65

#========== SCIPY LIBRARY ==========
#Z-Scores          : [-1.36 -0.53  0.43 -0.17  1.63]
#Mode              : 65

#========== PANDAS LIBRARY ==========

#Student Data with Average:
#Student  Maths  Python  AI   Average
#   Amit     78      82  80 80.000000
#   Neha     85      90  88 87.666667
#  Rahul     72      75  70 72.333333
#  Sneha     90      95  92 92.333333
#  Priya     88      86  90 88.000000

#Summary Statistics:
#       Maths  Python     AI  Average
#count   5.00    5.00   5.00     5.00
#mean   82.60   85.60  84.00    84.07
#std     7.47    7.64   9.06     7.92
#min    72.00   75.00  70.00    72.33
#25%    78.00   82.00  80.00    80.00
#50%    85.00   86.00  88.00    87.67
#75%    88.00   90.00  90.00    88.00
#max    90.00   95.00  92.00    92.33
