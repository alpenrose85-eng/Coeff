import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Настройка страницы
st.set_page_config(page_title="Подбор коэффициента жаропрочности", layout="wide")
st.title("Подбор оптимального коэффициента C для параметров жаропрочности")

# Выбор модели
model = st.radio(
    "Выберите параметр жаропрочности:",
    ("Ларсона–Миллера", "Трунина"),
    horizontal=True
)

# Ввод данных
st.subheader("Введите экспериментальные данные")
st.markdown("Добавьте строки с помощью кнопки ниже. Укажите:")
st.markdown("- **Температура** в Кельвинах (K)")
st.markdown("- **Время до разрушения** в часах (τ)")
st.markdown("- **Напряжение** в МПа")

# Инициализация данных
if "data" not in st.session_state:
    st.session_state.data = []

# Кнопка добавления строки
if st.button("Добавить строку"):
    st.session_state.data.append({"T": 1000, "tau": 1000, "sigma": 100})

# Отображение таблицы для ввода
edited_data = []
for i, row in enumerate(st.session_state.data):
    cols = st.columns(3)
    T = cols[0].number_input(f"Температура (K) {i+1}", value=float(row["T"]), min_value=300.0, step=10.0, key=f"T_{i}")
    tau = cols[1].number_input(f"Время (ч) {i+1}", value=float(row["tau"]), min_value=1.0, step=10.0, key=f"tau_{i}")
    sigma = cols[2].number_input(f"Напряжение (МПа) {i+1}", value=float(row["sigma"]), min_value=1.0, step=1.0, key=f"sigma_{i}")
    edited_data.append({"T": T, "tau": tau, "sigma": sigma})

st.session_state.data = edited_data

# Кнопка удаления последней строки
if st.button("Удалить последнюю строку") and st.session_state.data:
    st.session_state.data.pop()

# Преобразуем в DataFrame
df = pd.DataFrame(st.session_state.data)
if df.empty or len(df) < 3:
    st.warning("Добавьте минимум 3 точки для корректного анализа.")
    st.stop()

# Проверка на корректность данных
if (df["T"] <= 0).any() or (df["tau"] <= 0).any() or (df["sigma"] <= 0).any():
    st.error("Все значения должны быть положительными!")
    st.stop()

# Функции для вычисления параметров
def calc_lm(T, tau, C):
    return T * (np.log10(tau) + C)

def calc_trunin(T, tau, C):
    return T * (np.log10(tau) - 2 * np.log10(T) + C)

# Функция для оценки качества подбора C
def objective(C, T, tau, sigma, model_func):
    P = model_func(T, tau, C)
    # Регрессия: log(sigma) = a * P + b
    log_sigma = np.log10(sigma)
    reg = LinearRegression().fit(P.reshape(-1, 1), log_sigma)
    pred = reg.predict(P.reshape(-1, 1))
    r2 = r2_score(log_sigma, pred)
    # Минимизируем отрицательный R² → максимизируем R²
    return -r2

# Выбор функции модели
model_func = calc_trunin if model == "Трунина" else calc_lm

# Оптимизация C
T_vals = df["T"].values
tau_vals = df["tau"].values
sigma_vals = df["sigma"].values

# Поиск оптимального C в разумном диапазоне
res = minimize_scalar(
    objective,
    bounds=(10, 30) if model == "Ларсона–Миллера" else (-5, 15),
    args=(T_vals, tau_vals, sigma_vals, model_func),
    method='bounded'
)

C_opt = res.x
P_opt = model_func(T_vals, tau_vals, C_opt)
log_sigma = np.log10(sigma_vals)

# Линейная регрессия
reg = LinearRegression().fit(P_opt.reshape(-1, 1), log_sigma)
pred_log_sigma = reg.predict(P_opt.reshape(-1, 1))
r2 = r2_score(log_sigma, pred_log_sigma)

# Вывод результатов
st.subheader("Результаты подбора")
col1, col2 = st.columns(2)
col1.metric("Оптимальный коэффициент C", f"{C_opt:.4f}")
col2.metric("Коэффициент детерминации R²", f"{r2:.4f}")

# Уравнение регрессии
a = reg.coef_[0]
b = reg.intercept_
st.markdown(f"**Уравнение регрессии:** log₁₀(σ) = {a:.4f} · P + {b:.4f}")

# График
st.subheader("График зависимости напряжения от параметра жаропрочности")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(P_opt, sigma_vals, color='red', label='Экспериментальные данные')
# Построим плавную кривую регрессии
P_fit = np.linspace(P_opt.min(), P_opt.max(), 100)
sigma_fit = 10 ** (a * P_fit + b)
ax.plot(P_fit, sigma_fit, 'b--', label='Аппроксимация')
ax.set_xlabel("Параметр жаропрочности P")
ax.set_ylabel("Напряжение, МПа")
ax.set_yscale('log')
ax.grid(True, which="both", ls="--", lw=0.5)
ax.legend()
st.pyplot(fig)

# Таблица с рассчитанными параметрами
st.subheader("Рассчитанные значения параметра")
df_result = df.copy()
df_result["P"] = P_opt
df_result["log10(σ)"] = log_sigma
st.dataframe(df_result.round(4))