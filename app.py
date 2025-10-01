import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import json

# Настройка страницы
st.set_page_config(page_title="Параметры жаропрочности", layout="wide")
st.title("Подбор коэффициента C для параметров жаропрочности")

# === ЗАГРУЗКА ПРОЕКТА ===
st.sidebar.subheader("Сохранение / загрузка")
uploaded_file = st.sidebar.file_uploader("Загрузите проект (.json)", type=["json"])
if uploaded_file is not None:
    try:
        data_from_file = json.load(uploaded_file)
        st.session_state.data = data_from_file.get("data", [])
        st.session_state.model = data_from_file.get("model", "Ларсона–Миллера")
    except Exception as e:
        st.sidebar.error(f"Ошибка загрузки: {e}")

# === ВЫБОР МОДЕЛИ ===
model_options = ["Ларсона–Миллера", "Трунина"]
if "model" not in st.session_state:
    st.session_state.model = model_options[0]

model = st.radio(
    "Выберите параметр жаропрочности:",
    model_options,
    index=model_options.index(st.session_state.model),
    key="model_radio"
)
st.session_state.model = model

# === ВВОД ДАННЫХ ===
st.subheader("Введите экспериментальные данные")
st.markdown("""
- **Температура** в градусах Цельсия (°C)  
- **Время до разрушения** в часах (τ)  
- **Напряжение** в МПа
""")

if "data" not in st.session_state:
    st.session_state.data = []

# Кнопки управления
col_btn1, col_btn2, col_btn3 = st.columns(3)
if col_btn1.button("Добавить строку"):
    st.session_state.data.append({"T_C": 727.0, "tau": 1000.0, "sigma": 100.0})  # 727°C ≈ 1000 K
if col_btn2.button("Удалить последнюю"):
    if st.session_state.data:
        st.session_state.data.pop()
if col_btn3.button("Очистить всё"):
    st.session_state.data = []

# Таблица ввода (в °C)
edited_data = []
for i, row in enumerate(st.session_state.data):
    cols = st.columns(3)
    T_C = cols[0].number_input(f"T (°C) {i+1}", value=float(row["T_C"]), min_value=100.0, max_value=1500.0, step=10.0, key=f"T_C_{i}")
    tau = cols[1].number_input(f"τ (ч) {i+1}", value=float(row["tau"]), min_value=1.0, step=10.0, key=f"tau_{i}")
    sigma = cols[2].number_input(f"σ (МПа) {i+1}", value=float(row["sigma"]), min_value=1.0, step=1.0, key=f"sigma_{i}")
    edited_data.append({"T_C": T_C, "tau": tau, "sigma": sigma})

st.session_state.data = edited_data

# === ПРЕОБРАЗОВАНИЕ В КЕЛЬВИНЫ И ПРОВЕРКА ===
df = pd.DataFrame(st.session_state.data)
if df.empty or len(df) < 3:
    st.warning("Добавьте минимум 3 точки для анализа.")
    st.stop()

if (df["tau"] <= 0).any() or (df["sigma"] <= 0).any():
    st.error("Время и напряжение должны быть положительными!")
    st.stop()

# Перевод в Кельвины
df["T_K"] = df["T_C"] + 273.15

# === ФОРМУЛЫ ПАРАМЕТРОВ (в Кельвинах!) ===
def calc_lm(T_K, tau, C):
    """Параметр Ларсона–Миллера: P = T * (log10(tau) + C)"""
    return T_K * (np.log10(tau) + C)

def calc_trunin(T_K, tau, C):
    """Параметр Трунина: P = T * (log10(tau) - 2*log10(T) + C)"""
    return T_K * (np.log10(tau) - 2 * np.log10(T_K) + C)

# === ФУНКЦИЯ ОПТИМИЗАЦИИ ===
def objective(C, T_K, tau, sigma, model_func):
    P = model_func(T_K, tau, C)
    if not np.all(np.isfinite(P)) or np.any(np.isnan(P)):
        return 1e6
    log_sigma = np.log10(sigma)
    reg = LinearRegression().fit(P.reshape(-1, 1), log_sigma)
    pred = reg.predict(P.reshape(-1, 1))
    r2 = r2_score(log_sigma, pred)
    return -r2  # минимизируем → максимизируем R²

# === ВЫБОР МОДЕЛИ И ПОДБОР C ===
model_func = calc_trunin if model == "Трунина" else calc_lm

# Диапазоны C на основе инженерной практики
if model == "Ларсона–Миллера":
    C_bounds = (15, 25)
    formula_str = r"$P_{\text{ЛМ}} = T \cdot (\log_{10} \tau + C)$"
else:
    C_bounds = (5, 15)
    formula_str = r"$P_{\text{тр}} = T \cdot (\log_{10} \tau - 2 \log_{10} T + C)$"

T_K_vals = df["T_K"].values
tau_vals = df["tau"].values
sigma_vals = df["sigma"].values

res = minimize_scalar(
    objective,
    bounds=C_bounds,
    args=(T_K_vals, tau_vals, sigma_vals, model_func),
    method='bounded'
)

C_opt = res.x
P_opt = model_func(T_K_vals, tau_vals, C_opt)

if not np.all(np.isfinite(P_opt)):
    st.error("Ошибка расчёта параметра. Проверьте данные (особенно температуру и время).")
    st.stop()

# Регрессия
log_sigma = np.log10(sigma_vals)
reg = LinearRegression().fit(P_opt.reshape(-1, 1), log_sigma)
r2 = r2_score(log_sigma, reg.predict(P_opt.reshape(-1, 1)))
a, b = reg.coef_[0], reg.intercept_

# === ВЫВОД РЕЗУЛЬТАТОВ ===
st.subheader("Результаты подбора")
st.markdown(f"**Используемая формула:** {formula_str}")
st.markdown("> ⚠️ **T — температура в Кельвинах** (автоматически рассчитана из °C)")

col1, col2 = st.columns(2)
col1.metric("Оптимальный коэффициент C", f"{C_opt:.4f}")
col2.metric("Коэффициент детерминации R²", f"{r2:.4f}")

st.markdown(f"**Уравнение регрессии:** $\\log_{{10}}(\\sigma) = {a:.4f} \\cdot P + {b:.4f}$")

# === ГРАФИК ===
st.subheader("Зависимость напряжения от параметра жаропрочности")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(P_opt, sigma_vals, color='red', label='Эксперимент')
P_fit = np.linspace(P_opt.min(), P_opt.max(), 200)
sigma_fit = 10 ** (a * P_fit + b)
ax.plot(P_fit, sigma_fit, 'b--', label='Аппроксимация')
ax.set_xlabel("Параметр жаропрочности P")
ax.set_ylabel("Напряжение, МПа")
ax.set_yscale('log')
ax.grid(True, which="both", ls="--", alpha=0.6)
ax.legend()
st.pyplot(fig)

# === ТАБЛИЦА ===
st.subheader("Рассчитанные значения")
df_display = df[["T_C", "tau", "sigma"]].copy()
df_display["T (K)"] = df["T_K"]
df_display["P"] = P_opt
df_display["log10(σ)"] = log_sigma
df_display = df_display.rename(columns={"T_C": "T (°C)"})
st.dataframe(df_display.round(4))

# === СОХРАНЕНИЕ ПРОЕКТА ===
st.sidebar.markdown("---")
st.sidebar.subheader("Сохранить проект")
project_data = {
    "model": st.session_state.model,
    "data": st.session_state.data  # хранится T_C, как ввёл пользователь
}
json_str = json.dumps(project_data, indent=2)
st.sidebar.download_button(
    label="📥 Скачать проект (.json)",
    data=json_str,
    file_name="creep_project.json",
    mime="application/json"
)
