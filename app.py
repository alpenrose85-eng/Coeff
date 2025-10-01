import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import json
import io

# Настройка страницы
st.set_page_config(page_title="Параметры жаропрочности", layout="wide")
st.title("Подбор коэффициента C для параметров жаропрочности")

# === ЗАГРУЗКА ПРОЕКТА (JSON) ===
st.sidebar.subheader("Сохранение / загрузка проекта")
uploaded_project = st.sidebar.file_uploader("Загрузите проект (.json)", type=["json"])
if uploaded_project is not None:
    try:
        data_from_file = json.load(uploaded_project)
        st.session_state.data = data_from_file.get("data", [])
        st.session_state.model = data_from_file.get("model", "Ларсона–Миллера")
    except Exception as e:
        st.sidebar.error(f"Ошибка загрузки проекта: {e}")

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

# === МАССОВЫЙ ВВОД ДАННЫХ ===
st.subheader("Массовый ввод данных")

# Вариант 1: Загрузка файла
st.markdown("### 📎 Загрузите файл")
uploaded_file = st.file_uploader(
    "Поддерживаемые форматы: .xlsx, .xls, .csv",
    type=["xlsx", "xls", "csv"]
)

# Вариант 2: Вставка таблицы
st.markdown("### 📋 Или вставьте данные из Excel / Word")
st.markdown("""
Скопируйте диапазон ячеек (столбцы: **T (°C)**, **tau (ч)**, **sigma (МПа)**) и вставьте ниже.  
Разделители: **табуляция, запятая, пробел или точка с запятой**.
""")

pasted_data = st.text_area(
    "Вставьте данные (первая строка — заголовки, например: T_C, tau, sigma)",
    height=200,
    placeholder="T_C\tau\tsigma\n727\t1000\t120\n800\t500\t150"
)

# Обработка данных
df_input = None

# Приоритет: файл > вставка > сессия
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df_input = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        st.success(f"Загружено {len(df_input)} строк из файла.")
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")

elif pasted_data.strip():
    try:
        # Определяем разделитель
        sample = pasted_data.split('\n')[0]
        if '\t' in sample:
            sep = '\t'
        elif ';' in sample:
            sep = ';'
        elif ',' in sample:
            sep = ','
        else:
            sep = None  # пробелы или смешанные — pandas попробует сам

        from io import StringIO
        df_input = pd.read_csv(StringIO(pasted_data), sep=sep)
        st.success(f"Распознано {len(df_input)} строк из вставленных данных.")
    except Exception as e:
        st.error(f"Ошибка парсинга вставленных данных: {e}")

# Если данные получены — сохраняем в сессию
if df_input is not None:
    # Приводим названия колонок к единому виду
    col_map = {}
    for col in df_input.columns:
        col_lower = col.strip().lower()
        if col_lower in ["t", "t_c", "temperature", "температура", "t (°c)", "t(c)"]:
            col_map[col] = "T_C"
        elif col_lower in ["tau", "t_r", "время", "time", "τ", "tau (h)", "tau (ч)"]:
            col_map[col] = "tau"
        elif col_lower in ["sigma", "напряжение", "stress", "σ", "sigma (mpa)", "sigma (мпа)"]:
            col_map[col] = "sigma"
    
    df_input = df_input.rename(columns=col_map)
    
    required_cols = {"T_C", "tau", "sigma"}
    if not required_cols.issubset(df_input.columns):
        missing = required_cols - set(df_input.columns)
        st.error(f"Не хватает столбцов: {missing}. Убедитесь, что есть T_C, tau, sigma.")
    else:
        # Сохраняем только нужные столбцы как список словарей
        st.session_state.data = df_input[["T_C", "tau", "sigma"]].to_dict('records')

# === РУЧНОЙ ВВОД (оставим для небольших правок) ===
st.markdown("---")
st.subheader("Ручной ввод (для правки)")

if "data" not in st.session_state:
    st.session_state.data = []

# Кнопки управления
col_btn1, col_btn2, col_btn3 = st.columns(3)
if col_btn1.button("Добавить строку"):
    st.session_state.data.append({"T_C": 727.0, "tau": 1000.0, "sigma": 100.0})
if col_btn2.button("Удалить последнюю"):
    if st.session_state.
        st.session_state.data.pop()
if col_btn3.button("Очистить всё"):
    st.session_state.data = []

# Таблица ручного ввода
edited_data = []
for i, row in enumerate(st.session_state.data):
    cols = st.columns(3)
    T_C = cols[0].number_input(f"T (°C) {i+1}", value=float(row["T_C"]), min_value=100.0, max_value=1500.0, step=10.0, key=f"T_C_{i}")
    tau = cols[1].number_input(f"τ (ч) {i+1}", value=float(row["tau"]), min_value=1.0, step=10.0, key=f"tau_{i}")
    sigma = cols[2].number_input(f"σ (МПа) {i+1}", value=float(row["sigma"]), min_value=1.0, step=1.0, key=f"sigma_{i}")
    edited_data.append({"T_C": T_C, "tau": tau, "sigma": sigma})

st.session_state.data = edited_data

# === ПРОВЕРКА И РАСЧЁТ ===
df = pd.DataFrame(st.session_state.data)
if df.empty or len(df) < 3:
    st.warning("Добавьте минимум 3 точки для анализа.")
    st.stop()

if (df["tau"] <= 0).any() or (df["sigma"] <= 0).any():
    st.error("Время и напряжение должны быть положительными!")
    st.stop()

df["T_K"] = df["T_C"] + 273.15

# === ФУНКЦИИ ПАРАМЕТРОВ ===
def calc_lm(T_K, tau, C):
    return T_K * (np.log10(tau) + C)

def calc_trunin(T_K, tau, C):
    return T_K * (np.log10(tau) - 2 * np.log10(T_K) + C)

def objective(C, T_K, tau, sigma, model_func):
    P = model_func(T_K, tau, C)
    if not np.all(np.isfinite(P)):
        return 1e6
    log_sigma = np.log10(sigma)
    reg = LinearRegression().fit(P.reshape(-1, 1), log_sigma)
    r2 = r2_score(log_sigma, reg.predict(P.reshape(-1, 1)))
    return -r2

# === ВЫБОР МОДЕЛИ ===
model_func = calc_trunin if model == "Трунина" else calc_lm
if model == "Ларсона–Миллера":
    C_bounds = (15, 25)
    formula_str = r"$P_{\text{ЛМ}} = T \cdot (\log_{10} \tau + C)$"
else:
    C_bounds = (5, 15)
    formula_str = r"$P_{\text{тр}} = T \cdot (\log_{10} \tau - 2 \log_{10} T + C)$"

T_K_vals = df["T_K"].values
tau_vals = df["tau"].values
sigma_vals = df["sigma"].values

res = minimize_scalar(objective, bounds=C_bounds, args=(T_K_vals, tau_vals, sigma_vals, model_func), method='bounded')
C_opt = res.x
P_opt = model_func(T_K_vals, tau_vals, C_opt)

if not np.all(np.isfinite(P_opt)):
    st.error("Ошибка расчёта параметра. Проверьте данные.")
    st.stop()

log_sigma = np.log10(sigma_vals)
reg = LinearRegression().fit(P_opt.reshape(-1, 1), log_sigma)
r2 = r2_score(log_sigma, reg.predict(P_opt.reshape(-1, 1)))
a, b = reg.coef_[0], reg.intercept_

# === ВЫВОД ===
st.subheader("Результаты подбора")
st.markdown(f"**Формула:** {formula_str}")
st.markdown("> T — температура в Кельвинах (T = T°C + 273.15)")

col1, col2 = st.columns(2)
col1.metric("C", f"{C_opt:.4f}")
col2.metric("R²", f"{r2:.4f}")

st.markdown(f"**Регрессия:** $\\log_{{10}}(\\sigma) = {a:.4f} \\cdot P + {b:.4f}$")

# График
st.subheader("График")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(P_opt, sigma_vals, color='red', label='Данные')
P_fit = np.linspace(P_opt.min(), P_opt.max(), 200)
sigma_fit = 10 ** (a * P_fit + b)
ax.plot(P_fit, sigma_fit, 'b--', label='Аппроксимация')
ax.set_xlabel("Параметр P")
ax.set_ylabel("Напряжение, МПа")
ax.set_yscale('log')
ax.grid(True, which="both", ls="--", alpha=0.6)
ax.legend()
st.pyplot(fig)

# Таблица
st.subheader("Данные")
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
    "data": st.session_state.data
}
json_str = json.dumps(project_data, indent=2)
st.sidebar.download_button(
    label="📥 Скачать проект (.json)",
    data=json_str,
    file_name="creep_project.json",
    mime="application/json"
)
