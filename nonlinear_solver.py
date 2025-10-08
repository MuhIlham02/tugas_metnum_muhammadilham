import numpy as np
import pandas as pd
from tabulate import tabulate

# Sistem persamaan:
# f1(x,y) = x^2 + xy - 10 = 0
# f2(x,y) = y + 3xy^2 - 57 = 0

# Initial values
x0, y0 = 1.5, 3.5
epsilon = 0.000001
max_iter = 100

print("="*80)
print("PENYELESAIAN SISTEM PERSAMAAN NON-LINEAR")
print("NIM: 21120123120003 (NIMx = 3)")
print("="*80)
print(f"f1(x,y) = x² + xy - 10 = 0")
print(f"f2(x,y) = y + 3xy² - 57 = 0")
print(f"Initial: x0 = {x0}, y0 = {y0}, epsilon = {epsilon}")
print("="*80)

# Fungsi-fungsi persamaan
def f1(x, y):
    return x**2 + x*y - 10

def f2(x, y):
    return y + 3*x*y**2 - 57

# Fungsi iterasi g1B dan g2B (NIMx = 3)
# g1B: x = sqrt(10 - xy) atau x = (10 - xy)^0.5
# g2B: y = (57 - y)/(3xy) atau y = 57/(1 + 3xy)

def g1B(x, y):
    """x = sqrt(10 - xy)"""
    val = 10 - x*y
    if val < 0:
        return None
    return np.sqrt(val)

def g2B(x, y):
    """y = (57 - y)/(3xy)"""
    denom = 1 + 3*x*y
    if abs(denom) < 1e-10:
        return None
    return 57 / denom

# Turunan parsial untuk Newton-Raphson
def df1_dx(x, y):
    return 2*x + y

def df1_dy(x, y):
    return x

def df2_dx(x, y):
    return 3*y**2

def df2_dy(x, y):
    return 1 + 6*x*y

# METODE 1: ITERASI TITIK TETAP - JACOBI dengan g1B dan g2B
print("\n" + "="*80)
print("METODE 1: ITERASI TITIK TETAP - JACOBI (g1B dan g2B)")
print("="*80)

x, y = x0, y0
results_jacobi = []
converged = False

for i in range(max_iter):
    x_new = g1B(x, y)
    y_new = g2B(x, y)
    
    if x_new is None or y_new is None:
        print("Error: Fungsi iterasi tidak terdefinisi")
        break
    
    error = max(abs(x_new - x), abs(y_new - y))
    results_jacobi.append([i, x, y, f1(x, y), f2(x, y), error])
    
    if error < epsilon:
        x, y = x_new, y_new
        results_jacobi.append([i+1, x, y, f1(x, y), f2(x, y), error])
        converged = True
        print(f"Konvergen pada iterasi {i+1}")
        break
    
    x, y = x_new, y_new

df_jacobi = pd.DataFrame(results_jacobi, 
                         columns=['Iterasi', 'x', 'y', 'f1(x,y)', 'f2(x,y)', 'Error'])
print(tabulate(df_jacobi.tail(10), headers='keys', tablefmt='grid', floatfmt='.8f'))
if converged:
    print(f"\nSolusi: x = {x:.8f}, y = {y:.8f}")

# METODE 2: ITERASI TITIK TETAP - SEIDEL dengan g1B dan g2B
print("\n" + "="*80)
print("METODE 2: ITERASI TITIK TETAP - GAUSS-SEIDEL (g1B dan g2B)")
print("="*80)

x, y = x0, y0
results_seidel = []
converged = False

for i in range(max_iter):
    x_old, y_old = x, y
    
    x_new = g1B(x, y)
    if x_new is None:
        print("Error: Fungsi iterasi g1B tidak terdefinisi")
        break
    x = x_new
    
    y_new = g2B(x, y)
    if y_new is None:
        print("Error: Fungsi iterasi g2B tidak terdefinisi")
        break
    y = y_new
    
    error = max(abs(x - x_old), abs(y - y_old))
    results_seidel.append([i, x, y, f1(x, y), f2(x, y), error])
    
    if error < epsilon:
        converged = True
        print(f"Konvergen pada iterasi {i+1}")
        break

df_seidel = pd.DataFrame(results_seidel, 
                         columns=['Iterasi', 'x', 'y', 'f1(x,y)', 'f2(x,y)', 'Error'])
print(tabulate(df_seidel.tail(10), headers='keys', tablefmt='grid', floatfmt='.8f'))
if converged:
    print(f"\nSolusi: x = {x:.8f}, y = {y:.8f}")

# METODE 3: NEWTON-RAPHSON
print("\n" + "="*80)
print("METODE 3: NEWTON-RAPHSON")
print("="*80)

x, y = x0, y0
results_nr = []
converged = False

for i in range(max_iter):
    # Jacobian matrix
    J = np.array([[df1_dx(x, y), df1_dy(x, y)],
                  [df2_dx(x, y), df2_dy(x, y)]])
    
    # Function values
    F = np.array([f1(x, y), f2(x, y)])
    
    # Solve J * delta = -F
    try:
        delta = np.linalg.solve(J, -F)
    except:
        print("Error: Jacobian singular")
        break
    
    x_new = x + delta[0]
    y_new = y + delta[1]
    
    error = max(abs(delta[0]), abs(delta[1]))
    results_nr.append([i, x, y, f1(x, y), f2(x, y), error])
    
    if error < epsilon:
        x, y = x_new, y_new
        results_nr.append([i+1, x, y, f1(x, y), f2(x, y), error])
        converged = True
        print(f"Konvergen pada iterasi {i+1}")
        break
    
    x, y = x_new, y_new

df_nr = pd.DataFrame(results_nr, 
                     columns=['Iterasi', 'x', 'y', 'f1(x,y)', 'f2(x,y)', 'Error'])
print(tabulate(df_nr, headers='keys', tablefmt='grid', floatfmt='.8f'))
if converged:
    print(f"\nSolusi: x = {x:.8f}, y = {y:.8f}")

# METODE 4: SECANT
print("\n" + "="*80)
print("METODE 4: SECANT")
print("="*80)

# Untuk Secant, kita perlu 2 tebakan awal untuk setiap variabel
x0_sec, y0_sec = 1.5, 3.5
x1_sec, y1_sec = 1.6, 3.6

results_secant = []
converged = False

x_prev, y_prev = x0_sec, y0_sec
x_curr, y_curr = x1_sec, y1_sec

for i in range(max_iter):
    f1_prev = f1(x_prev, y_prev)
    f2_prev = f2(x_prev, y_prev)
    f1_curr = f1(x_curr, y_curr)
    f2_curr = f2(x_curr, y_curr)
    
    # Approximate Jacobian using secant method
    if abs(x_curr - x_prev) < 1e-10 or abs(y_curr - y_prev) < 1e-10:
        print("Error: Pembagi terlalu kecil")
        break
    
    # Simple secant update for each variable
    denom_x = f1_curr - f1_prev
    denom_y = f2_curr - f2_prev
    
    if abs(denom_x) < 1e-10 or abs(denom_y) < 1e-10:
        print("Error: Denominador terlalu kecil")
        break
    
    x_new = x_curr - f1_curr * (x_curr - x_prev) / denom_x
    y_new = y_curr - f2_curr * (y_curr - y_prev) / denom_y
    
    error = max(abs(x_new - x_curr), abs(y_new - y_curr))
    results_secant.append([i, x_curr, y_curr, f1_curr, f2_curr, error])
    
    if error < epsilon:
        results_secant.append([i+1, x_new, y_new, f1(x_new, y_new), f2(x_new, y_new), error])
        converged = True
        print(f"Konvergen pada iterasi {i+1}")
        break
    
    x_prev, y_prev = x_curr, y_curr
    x_curr, y_curr = x_new, y_new

df_secant = pd.DataFrame(results_secant, 
                         columns=['Iterasi', 'x', 'y', 'f1(x,y)', 'f2(x,y)', 'Error'])
print(tabulate(df_secant.tail(10), headers='keys', tablefmt='grid', floatfmt='.8f'))
if converged:
    print(f"\nSolusi: x = {x_new:.8f}, y = {y_new:.8f}")

# RINGKASAN
print("\n" + "="*80)
print("RINGKASAN HASIL")
print("="*80)
print("\nPerbandingan Metode:")
print(f"1. Jacobi (g1B, g2B)      : {len(results_jacobi)} iterasi")
print(f"2. Gauss-Seidel (g1B, g2B): {len(results_seidel)} iterasi")
print(f"3. Newton-Raphson         : {len(results_nr)} iterasi")
print(f"4. Secant                 : {len(results_secant)} iterasi")

# Export to Excel
with pd.ExcelWriter('hasil_iterasi_21120123120003.xlsx') as writer:
    df_jacobi.to_excel(writer, sheet_name='Jacobi', index=False)
    df_seidel.to_excel(writer, sheet_name='Seidel', index=False)
    df_nr.to_excel(writer, sheet_name='Newton-Raphson', index=False)
    df_secant.to_excel(writer, sheet_name='Secant', index=False)

print("\n✓ File Excel berhasil dibuat: hasil_iterasi_21120123120003.xlsx")