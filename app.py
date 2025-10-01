import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import json
from io import StringIO

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
Поддерживаются **запятые как десятичный разделитель** (например: 65,9).
""")

pasted_data = st.text_area(
    "Вставьте данные (первая строка — заголовки)",
    height=200,
    placeholder="T_C\tau\tsigma\n727\t1000\t65,9\n800\t500\t150"
)

# Обработка данных
df_input = None

# Загрузка файла
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df_input = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            # Пробуем сначала с точкой, потом с запятой
            try:
                df_input = pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                df_input = pd.read_csv(uploaded_file, decimal=',', sep=';')
        st.success(f"Загружено {len(df_input)} строк из файла.")
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")

# Вставка текста
elif pasted_data.strip():
    try:
        # Заменяем запятые-десятичные на точки
        cleaned_data = pasted_data.replace(",", ".")
        
        # Определяем разделитель
        first_line = cleaned_data.split('\n')[0]
        if '\t' in first_line:
            sep = '\t'
        elif ';' in first_line:
            sep = ';'
        elif first_line.count(' ') >= 2:
            sep = r'\s+'
        else:
            sep = None
        
        df_input = pd.read_csv(StringIO(cleaned_data), sep=sep, engine='python')
        st.success(f"Распознано {len(df_input)} строк из вставленных данных.")
    except Exception as e:
        st.error(f"Ошибка парсинга вставленных данных: {e}")

# Сохранение данных в сессию
if df_input is not None:
    # Приводим названия колонок к стандарту
    col_map = {}
    for col in df_input.columns:
        col_clean = str(col).strip().lower()
        if any(kw in col_clean for kw in ["t_c", "temp", "temperature", "температура", "t°c", "tc", "t (c)"]):
            col_map[col] = "T_C"
        elif any(kw in col_clean for kw in ["tau", "time", "время", "t_r", "τ", "час", "ч"]):
            col_map[col] = "tau"
        elif any(kw in col_clean for kw in ["sigma", "stress", "напряжение", "σ", "mpa", "мпа"]):
            col_map[col] = "sigma"
    
    df_input = df_input.rename(columns=col_map)
    
    required_cols = {"T_C", "tau", "sigma"}
    if not required_cols.issubset(df_input.columns):
        missing = required_cols - set(df_input.columns)
        st.error(f"Не хватает столбцов: {missing}. Нужны: T_C, tau, sigma")
    else:
        # Приведение к float с защитой от строк
        records = []
        for _, row in df_input[["T_C", "tau", "sigma"]].iterrows():
            try:
                records.append({
                    "T_C": float(row["T_C"]),
                    "tau": float(row["tau"]),
                    "sigma": float(row["sigma"])
                })
            except (ValueError, TypeError):
                st.warning(f"Пропущена некорректная строка: {row.to_dict()}")
                continue
        st.session_state.data = records

# === РУЧНОЙ ВВОД ===
st.markdown("---")
st.subheader("Ручной ввод (для правки)")

if "data" not in st.session_state:
    st.session_state.data = []

# Кнопки управления
col_btn1, col_btn2, col_btn3 = st.columns(3)
if col_btn1.button("Добавить строку"):
    st.session_state.data.append({"T_C": 727.0, "tau": 1000.0, "sigma": 100.0})
if col_btn2.button("Удалить последнюю"):
    if len(st.session_state.data) > 0:
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
col1.metric("Коэффициент C", f"{C_opt:.4f}")
col2.metric("Коэффициент детерминации R²", f"{r2:.4f}")

st.markdown(f"**Уравнение регрессии:** $\\log_{{10}}(\\sigma) = {a:.4f} \\cdot P + {b:.4f}$")

# График
st.subheader("График зависимости напряжения от параметра жаропрочности")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(P_opt, sigma_vals, color='red', label='Экспериментальные данные')
P_fit = np.linspace(P_opt.min(), P_opt.max(), 200)
sigma_fit = 10 ** (a * P_fit + b)
ax.plot(P_fit, sigma_fit, 'b--', label='Аппроксимация')
ax.set_xlabel("Параметр жаропрочности P")
ax.set_ylabel("Напряжение, МПа")
ax.set_yscale('log')
ax.grid(True, which="both", ls="--", alpha=0.6)
ax.legend()
st.pyplot(fig)

# Таблица результатов
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
    "data": st.session_state.data
}
json_str = json.dumps(project_data, indent=2)
st.sidebar.download_button(
    label="📥 Скачать проект (.json)",
    data=json_str,
    file_name="creep_project.json",
    mime="application/json"
)
