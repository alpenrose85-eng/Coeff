import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import json

# Настройка страницы
st.set_page_config(page_title="Подбор коэффициента жаропрочности", layout="wide")
st.title("Подбор оптимального коэффициента C для параметров жаропрочности")

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
- **Температура** в Кельвинах (K)  
- **Время до разрушения** в часах (τ)  
- **Напряжение** в МПа
""")

if "data" not in st.session_state:
    st.session_state.data = []

# Кнопки управления
col_btn1, col_btn2, col_btn3 = st.columns(3)
if col_btn1.button("Добавить строку"):
    st.session_state.data.append({"T": 1000.0, "tau": 1000.0, "sigma": 100.0})
if col_btn2.button("Удалить последнюю"):
    if st.session_state.data:
        st.session_state.data.pop()
if col_btn3.button("Очистить всё"):
    st.session_state.data = []

# Таблица ввода
edited_data = []
for i, row in enumerate(st.session_state.data):
    cols = st.columns(3)
    T = cols[0].number_input(f"T (K) {i+1}", value=float(row["T"]), min_value=300.0, step=10.0, key=f"T_{i}")
    tau = cols[1].number_input(f"τ (ч) {i+1}", value=float(row["tau"]), min_value=1.0, step=10.0, key=f"tau_{i}")
    sigma = cols[2].number_input(f"σ (МПа) {i+1}", value=float(row["sigma"]), min_value=1.0, step=1.0, key=f"sigma_{i}")
    edited_data.append({"T": T, "tau": tau, "sigma": sigma})

st.session_state.data = edited_data

# === ПРОВЕРКА ДАННЫХ ===
df = pd.DataFrame(st.session_state.data)
if df.empty or len(df) < 3:
    st.warning("Добавьте минимум 3 точки для анализа.")
    st.stop()

if (df <= 0).any().any():
    st.error("Все значения должны быть положительными!")
    st.stop()

# === ФОРМУЛЫ ПАРАМЕТРОВ ===
def calc_lm(T, tau, C):
    """Параметр Ларсона–Миллера: P = T * (log10(tau) + C)"""
    return T * (np.log10(tau) + C)

def calc_trunin(T, tau, C):
    """Параметр Трунина: P = T * (log10(tau) - 2*log10(T) + C)"""
    return T * (np.log10(tau) - 2 * np.log10(T) + C)

# === ФУНКЦИЯ ОЦЕНКИ КАЧЕСТВА ===
def objective(C, T, tau, sigma, model_func):
    P = model_func(T, tau, C)
    # Исключаем NaN/inf
    if not np.all(np.isfinite(P)):
        return 1e6
    log_sigma = np.log10(sigma)
    reg = LinearRegression().fit(P.reshape(-1, 1), log_sigma)
    pred = reg.predict(P.reshape(-1, 1))
    r2 = r2_score(log_sigma, pred)
    return -r2  # минимизируем отрицательный R²

# === ВЫБОР МОДЕЛИ И ОПТИМИЗАЦИЯ ===
model_func = calc_trunin if model == "Трунина" else calc_lm

# Диапазоны для C (уточнены на основе практики)
if model == "Ларсона–Миллера":
    C_bounds = (15, 25)
    formula_str = r"$P = T \cdot (\log_{10} \tau + C)$"
else:
    C_bounds = (5, 15)  # Более узкий и физичный диапазон для Трунина
    formula_str = r"$P = T \cdot (\log_{10} \tau - 2 \log_{10} T + C)$"

T_vals = df["T"].values
tau_vals = df["tau"].values
sigma_vals = df["sigma"].values

res = minimize_scalar(
    objective,
    bounds=C_bounds,
    args=(T_vals, tau_vals, sigma_vals, model_func),
    method='bounded'
)

C_opt = res.x
P_opt = model_func(T_vals, tau_vals, C_opt)

# Проверка на валидность
if not np.all(np.isfinite(P_opt)):
    st.error("Ошибка расчёта параметра: проверьте входные данные (особенно температуру и время).")
    st.stop()

log_sigma = np.log10(sigma_vals)
reg = LinearRegression().fit(P_opt.reshape(-1, 1), log_sigma)
r2 = r2_score(log_sigma, reg.predict(P_opt.reshape(-1, 1)))

# === ВЫВОД РЕЗУЛЬТАТОВ ===
st.subheader("Результаты подбора")
st.markdown(f"**Используемая формула:** {formula_str}")
col1, col2 = st.columns(2)
col1.metric("Оптимальный коэффициент C", f"{C_opt:.4f}")
col2.metric("Коэффициент детерминации R²", f"{r2:.4f}")

a, b = reg.coef_[0], reg.intercept_
st.markdown(f"**Уравнение регрессии:** $\\log_{{10}}(\\sigma) = {a:.4f} \\cdot P + {b:.4f}$")

# === ГРАФИК ===
st.subheader("Зависимость напряжения от параметра жаропрочности")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(P_opt, sigma_vals, color='red', label='Данные')
P_fit = np.linspace(P_opt.min(), P_opt.max(), 200)
sigma_fit = 10 ** (a * P_fit + b)
ax.plot(P_fit, sigma_fit, 'b--', label='Аппроксимация')
ax.set_xlabel("Параметр жаропрочности P")
ax.set_ylabel("Напряжение, МПа")
ax.set_yscale('log')
ax.grid(True, which="both", ls="--", alpha=0.6)
ax.legend()
st.pyplot(fig)

# === ТАБЛИЦА РЕЗУЛЬТАТОВ ===
st.subheader("Рассчитанные значения")
df_res = df.copy()
df_res["P"] = P_opt
df_res["log10(σ)"] = log_sigma
st.dataframe(df_res.round(4))

# === СОХРАНЕНИЕ ПРОЕКТА ===
st.sidebar.markdown("---")
st.sidebar.subheader("Сохранить проект")
project_data = {
    "model": st.session_state.model,
    "data": st.session_state.data
}
json_str = json.dumps(project_data, indent=2)
st.sidebar.download_button(
    label="📥 Скачать проект (.json)",
    data=json_str,
    file_name="creep_project.json",
    mime="application/json"
)
