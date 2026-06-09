import os, numpy as np, pandas as pd, warnings, matplotlib.pyplot as plt
import neurokit2 as nk
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from pathlib import Path

warnings.filterwarnings("ignore")
# Seleccionamos base de datos y parámetros de ventana
BASE_DIR = Path(__file__).parent

DATA_DIR    = BASE_DIR / "data" / "ECGS"
ANSWERS_DIR = BASE_DIR / "data" / "respuestas"

FS = 128
WINDOW_DURATION_MIN = 5

WINDOW_SAMPLES = int(WINDOW_DURATION_MIN * 60 * FS)
USE_OVERLAP = True # Cambia a True para activarlo

if USE_OVERLAP:
  OVERLAP = 0.5
  STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))
else:
  STEP_SAMPLES = WINDOW_SAMPLES

# --- Extracción de features ---
def extract_features_from_segment(segment, fs):
  try:
    # ✅ Normalización por ventana
    segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)

    _, rpeaks = nk.ecg_peaks(segment, sampling_rate=fs, correct_artifacts=True)
    peaks = rpeaks["ECG_R_Peaks"]

    if len(peaks) < 30:
      return None

    f = {}
    f.update(nk.hrv_time(peaks, sampling_rate=fs))
    f.update(nk.hrv_frequency(peaks, sampling_rate=fs))
    f.update(nk.hrv_nonlinear(peaks, sampling_rate=fs))

    # Delineación Onda P
    _, waves = nk.ecg_delineate(segment, peaks, sampling_rate=fs, method="dwt")
    p_on, p_off = np.array(waves["ECG_P_Onsets"]), np.array(waves["ECG_P_Offsets"])
    dur = p_off - p_on
    if np.any(~np.isnan(dur)):
      f["P_Duration_Mean"] = np.nanmean(dur) / fs
      f["P_Duration_STD"] = np.nanstd(dur) / fs

    return pd.DataFrame(f, index=[0])
  except:
    return None
  
def create_flattened_feature_vector(record, data_dir):
  try: sig = np.loadtxt(os.path.join(data_dir, f"{record}.txt"), usecols=(0,))
  except: return None
  feats = []
  for start in range(0, len(sig) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
    f = extract_features_from_segment(sig[start:start+WINDOW_SAMPLES], FS)
    if f is not None: feats.append(f)
  if not feats: return None
  out = pd.concat(feats, axis=1)
  out.columns = [f"{c}_w{i+1}" for i,c in enumerate(out.columns)]
  return out

def create_feature_dataset(records):
  lst = Parallel(n_jobs=-1)(delayed(create_flattened_feature_vector)(r, DATA_DIR) for r in records)
  out = []
  for r,v in zip(records,lst):
    if v is not None:
      v["record"]=r
      out.append(v)
  return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# --- Entrenamiento GridSearch ---
def optimize_and_train_optuna(X, y, seed):

  import optuna

  # --- Preprocesamiento igual que antes ---
  imp = SimpleImputer(strategy='mean').fit(X)
  sca = StandardScaler().fit(imp.transform(X))
  X_scaled = sca.transform(imp.transform(X))

  # Selección de features (mantenemos k=40)
  sel = SelectKBest(f_classif, k=min(40, X.shape[1])).fit(X_scaled, y)
  X_sel = sel.transform(X_scaled)

  cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)

  # --- Definimos el objetivo de optimización ---
  def objective(trial):
    C = trial.suggest_loguniform("C", 1e-3, 1e3)
    gamma = trial.suggest_loguniform("gamma", 1e-4, 1)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    model = SVC(
      kernel="rbf",
      C=C,
      gamma=gamma,
      class_weight=class_weight,
      probability=True,
      random_state=seed
    )

    scores = []
    for train_idx, val_idx in cv.split(X_sel, y):
      model.fit(X_sel[train_idx], y[train_idx])
      pred = model.predict(X_sel[val_idx])

      from sklearn.metrics import   
      scores.append(f1_score(y[val_idx], pred))

    return np.mean(scores)

  # --- Ejecutar búsqueda Optuna ---
  study = optuna.create_study(direction="maximize")
  study.optimize(objective, n_trials=80, show_progress_bar=True)

  best_params = study.best_params
  print(f"[Seed {seed}] Best={best_params}, F1={study.best_value:.3f}")

  # --- Entrenar modelo final con los mejores parámetros ---
  best_model = SVC(
    kernel="rbf",
    C=best_params["C"],
    gamma=best_params["gamma"],
    class_weight=best_params["class_weight"],
    probability=True,
    random_state=seed
  )

  best_model.fit(X_sel, y)

  return best_model, imp, sca, sel, study.best_value


# --- Evaluación oficial tipo Evento-2 ---
def load_answers(path):
  try:
    with open(path) as f: return {p[0]:p[1] for l in f if (p:=l.strip().split())}
  except: return {}

def eval_event2(preds,a1,a2):
  raw,off=0,0
  for i in range(1,100,2):
    s=f"t{i:02d}"; st=a1.get(s,'N')
    real=[a2.get(f"t{i:02d}",'N/A'),a2.get(f"t{i+1:02d}",'N/A')]
    mine=[preds.get(f"t{i:02d}",'N/A'),preds.get(f"t{i+1:02d}",'N/A')]
    ok=(st=='N') or (st=='A' and mine==real)
    raw+=ok; off+=int(st=='A' and mine==real)
  nA=sum(1 for v in a1.values() if v=='A')
  return raw, off, off/nA if nA>0 else 0

# --- Bucle de 10 semillas ---
p_records=[f"p{i:02d}" for i in range(1,51)]
train_df = create_feature_dataset(p_records)
train_df["label"] = train_df["record"].apply(lambda x: 1 if int(x[1:])%2==0 else 0)

# ✅ Nuevo X y y con features Δ
X = train_df.drop(columns=["record", "label"])
y = train_df["label"]
X_diff = X.groupby(train_df["record"]).diff().add_suffix("_Δ")
X = pd.concat([X, X_diff.fillna(0)], axis=1)
test_records=[f"t{i:02d}" for i in range(1,101)]
test_df=create_feature_dataset(test_records)
X_test,records=test_df.drop(columns=["record"]),test_df["record"]

# --- Bucle de 20 semillas ---
results = []

for seed in range(10):
  print(f"\n=== Semilla {seed} ===")

  # 1) Entrenamiento SVM con esta seed
  model, imp, sca, sel, val_f1 = optimize_and_train_optuna(X, y, seed)

  # 2) Transformación del test con el mismo pipeline
  X_test = test_df.drop(columns=["record"])
  record_col = test_df["record"]

  # Crear Δ en test igual que en train
  X_test_diff = X_test.groupby(record_col).diff().add_suffix("_Δ")
  X_test = pd.concat([X_test, X_test_diff.fillna(0)], axis=1)

  # Reordenar columnas para que coincidan exactamente con las de entrenamiento  
  X_test = X_test.reindex(columns=X.columns, fill_value=0)

  # Aplicar pipeline del modelo (imputar → escalar → seleccionar features)
  X_test_imp = imp.transform(X_test)
  X_test_scaled = sca.transform(X_test_imp)
  X_test_sel = sel.transform(X_test_scaled)

  # Probabilidad final
  probs = model.predict_proba(X_test_sel)[:, 1]
  patient = {r: {'prob_imminent': p} for r, p in zip(records, probs)}

  # 4) Decisión por cada par (Evento 2)
  preds = {}
  for i in range(1, 100, 2):
    r1, r2 = f"t{i:02d}", f"t{i+1:02d}"
    s1, s2 = patient[r1]['prob_imminent'], patient[r2]['prob_imminent']
    preds[r1], preds[r2] = ('N', 'A') if s2 > s1 else ('A', 'N')

  # 5) Evaluación oficial
  a1 = load_answers(os.path.join(ANSWERS_DIR, "event-1-answers.txt"))
  a2 = load_answers(os.path.join(ANSWERS_DIR, "event-2-answers.txt"))
  raw, off, accA = eval_event2(preds, a1, a2)

  # Guardar resultados
  results.append({
    "seed": seed,
    "val_f1": val_f1,
    "precision_A": accA
  })

df = pd.DataFrame(results)

print("\n=== Resultados Evento 2 por Semilla ===")
print(df)
print(f"\nPromedio F1 = {df.val_f1.mean():.3f} ± {df.val_f1.std():.3f}")
print(f"Promedio Precisión Grupo A = {df.precision_A.mean():.3f} ± {df.precision_A.std():.3f}")