# Guide de lecture du code

Document de travail en français, destiné à comprendre le dépôt ligne à ligne. Il n'est pas
nécessaire au fonctionnement du projet — le README anglais est la vitrine, celui-ci est l'atelier.

---

## 0. Le modèle mental

Le projet répond à une question : *combien coûte, en accuracy, le fait de prendre un modèle plus
petit ?* Pour y répondre proprement il faut mesurer plusieurs modèles **dans des conditions
identiques**, puis décider si les écarts observés sont réels ou du bruit.

D'où trois étapes séparées, et une règle : **chaque étape écrit sur le disque, l'étape suivante
relit**. Rien n'est gardé en mémoire entre les étapes.

```
config.py                       ← tout le déclaratif : quels modèles, quel dataset, quels réglages
    │
    ▼
train.py                        ← fine-tune chaque modèle, CHRONOMÈTRE l'entraînement
    │  écrit : results/checkpoints/<tag>/          (les poids)
    │          results/train_<tag>.json            (temps, paramètres, hyper-paramètres)
    ▼
evaluate.py                     ← recharge chaque checkpoint, mesure accuracy + latence
    │  écrit : results/eval_<tag>.json             (accuracy, p50, p95, throughput, étendue)
    │          results/predictions/<tag>.npz       (juste/faux exemple par exemple)
    │          results/environment.json            (machine, versions, mise en régime)
    ▼
report.py                       ← joint tout, calcule les quantités relatives + les tests
       écrit : results/benchmark.csv               (une ligne par modèle, tout le brut)
               results/benchmark.md                (tableaux + analyse rédigée)
               results/analysis.json               (idem, lisible par machine)
               plots/*.png                         (3 figures × 2 modes)
```

Autour de ce tuyau, cinq modules de service :

| Module | Rôle | Analogie |
|---|---|---|
| `common.py` | données, métriques, chronos, statistiques, comptage de paramètres | la boîte à outils |
| `plotting.py` | construction des figures | l'atelier graphique |
| `run_benchmark.py` | enchaîne train → evaluate → report | le chef d'orchestre |
| `selftest.py` | vérifie stats + rapport + figures **sans** `transformers` | le filet de sécurité |
| `check_env.py` | diagnostic de l'environnement + conseils de dimensionnement | le contrôle technique |

**Pourquoi ce découpage ?** Trois raisons concrètes :

1. L'entraînement coûte 20 à 40 minutes. Si le rapport plantait à la fin, tout serait perdu.
   Là, `report.py` se relance en 3 secondes sur les JSON déjà écrits.
2. Le chronométrage doit être propre. `train.py` ne fait *que* de l'entraînement
   (`eval_strategy="no"`), sinon le temps mesuré contiendrait aussi des évaluations.
3. Les tests statistiques ont besoin de savoir, **exemple par exemple**, quel modèle a eu bon.
   D'où les `.npz` : ce sont eux qui rendent les tests appariés possibles.

Le mot **`tag`** revient partout. C'est l'identifiant d'un run :

```python
def run_tag(slug, dataset, seed):
    return f"{slug}__{dataset}__seed{seed}"      # → "distilbert__ag_news__seed42"
```

Il sert de nom de fichier. Trois runs sur trois seeds ne s'écrasent donc jamais.

Une exception à ce schéma : `--smoke`. Un run de fumée utilise **les mêmes tags** qu'un vrai run,
donc il écrit tout sous `results/smoke/` (checkpoints, records, prédictions, figures) au lieu de
`results/` et `plots/`. Sans cela, vérifier le pipeline après un vrai sweep remplaçait ses
résultats par des versions entraînées sur 200 exemples. Voir `config.smoke_dirs()` en §3.1.

---

## 1. Ordre de lecture conseillé

Ne lis pas les fichiers par ordre alphabétique. Cet ordre-là fonctionne :

| # | Fichier | Ce que tu y gagnes | Temps |
|---|---|---|---|
| 1 | `config.py` | le vocabulaire : `ModelSpec`, `DatasetSpec`, `BenchConfig` | 10 min |
| 2 | `train.py`, fonction `train_one` | la boucle principale, de bout en bout | 20 min |
| 3 | `evaluate.py`, la docstring du module puis `score_one`, `time_all`, `aggregate_timings` | le protocole de mesure | 30 min |
| 4 | `common.py`, section « Statistics » | le cœur mathématique | 30 min |
| 5 | `report.py`, fonction `analyse` | toutes les quantités dérivées | 30 min |
| 6 | `plotting.py` | seulement si les figures t'intéressent | 30 min |
| 7 | `run_benchmark.py`, `selftest.py`, `check_env.py` | de la plomberie, lisible en diagonale | 15 min |

**Astuce de lecture** : chaque fonction a une docstring qui dit *pourquoi*, pas *comment* — le
comment se lit dans le code. Si tu lis d'abord toutes les docstrings d'un fichier, tu as le plan
avant les détails. Dans VS Code : `Ctrl+Shift+O` liste les symboles du fichier courant.

---

## 2. Les idiomes Python qui reviennent

Le code utilise une dizaine de tournures modernes. Les voici une fois pour toutes.

### 2.1 `from __future__ import annotations`

Première ligne de chaque fichier. Elle rend les annotations de type **paresseuses** : Python ne
les évalue pas à l'exécution, il les garde comme du texte. Conséquence pratique : on peut écrire
`int | None` ou `list[dict]` même sur des versions de Python où ces syntaxes seraient refusées à
l'exécution, et on peut annoter avec une classe définie plus bas dans le fichier.

### 2.2 Les annotations de type

```python
def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
```

Ça se lit : `successes` est un entier, `n` un entier, `z` un flottant valant 1.9599… par défaut,
et la fonction renvoie un couple de flottants. **Python ne vérifie rien** — c'est de la
documentation exécutable, utilisée par l'éditeur pour l'autocomplétion et par les linters.

`int | None` = « un entier ou rien ». Très fréquent ici : `max_train_samples: int | None = 4000`
signifie « 4000 par défaut, `None` pour dire *tout le split* ».

### 2.3 `@dataclass` — les structures de données

```python
@dataclass(frozen=True)
class ModelSpec:
    slug: str
    checkpoint: str
    lr: float
```

Le décorateur écrit pour toi `__init__`, `__repr__`, `__eq__`. Tu déclares juste les champs.

- `frozen=True` → **immuable**. `spec.lr = 1e-4` lève une erreur. Utilisé pour `ModelSpec` et
  `DatasetSpec` : ce sont des faits sur les modèles, personne ne doit les modifier en cours de route.
- sans `frozen` → **modifiable**. C'est le cas de `BenchConfig`, justement parce que
  `build_config()` ajuste ses champs selon les arguments de la ligne de commande.

`dataclasses.asdict(cfg)` transforme la dataclass en dictionnaire — c'est comme ça que la
configuration finit dans le JSON.

### 2.4 Les compréhensions

```python
total = sum(p.numel() for p in model.parameters())                       # générateur
lows  = [100 * r["accuracy_ci95_low"] for r in rows]                     # liste
frontier = {rows[i]["slug"] for i in front_idx}                          # ensemble
cmap = {row["slug"]: c for row, c in zip(rows, colors)}                  # dictionnaire
```

Même schéma partout : `expression for variable in itérable`. Crochets → liste, accolades avec
`:` → dictionnaire, accolades sans `:` → ensemble, parenthèses (ou rien, comme argument unique) →
générateur paresseux.

`zip(a, b)` avance dans deux listes en parallèle :

```python
for pos, val, color in zip(positions, values, colors):
```

### 2.5 `*` et `**` — étoiles

Trois usages distincts dans ce dépôt.

**Déballer un dictionnaire en arguments nommés** :

```python
model(**inputs)          # équivaut à model(input_ids=..., attention_mask=...)
```

**Fusionner des dictionnaires dans un littéral** :

```python
record = {
    "slug": model_spec.slug,
    **params,                    # injecte params_total, params_embedding, …
    "train_time_s": wall,
}
```

C'est le mécanisme central des records. Dans `evaluate.py`, `score_one` empile `**metrics` et
`**params` dans un dictionnaire plat ; chaque mesure de chronométrage fait de même avec
`{**latency, **throughput, …}` ; et à la fin la ligne définitive est une simple fusion :

```python
record = {**item["record"], **timings}       # timings = aggregate_timings(rounds[name])
```

Ce dictionnaire plat devient une ligne du CSV. En cas de clé commune, c'est le dictionnaire de
**droite** qui gagne.

**Forcer les arguments nommés** — l'étoile nue dans une signature :

```python
def fig_pareto(rows, frontier, theme, *, latency_key: str, latency_label: str, cmap=None):
```

Tout ce qui suit le `*` doit être passé par son nom : `fig_pareto(rows, front, th,
latency_key="latency_ms_p50", …)`. On ne peut pas les passer par position. C'est volontaire :
`fig_pareto(rows, front, th, "latency_ms_p50", "Latence")` serait illisible.

**`nargs="+"` dans argparse** : `--models a b c` collecte une liste.

### 2.6 Les f-strings et leurs formats

```python
f"{100 * r['accuracy']:.2f}%"        # 89.95%      2 décimales
f"{row['delta_accuracy_pp']:+.2f}"   # -0.35       signe toujours affiché
f"{n_eval:,}"                        # 2,000       séparateur de milliers
f"{trunc:.1%}"                       # 0.4%        multiplie par 100 et ajoute %
f"{lr:g}"                            # 5e-05       notation « la plus courte »
f"{wall:6.1f}"                       # «  12.3 »   largeur 6, aligné à droite
f"{'=' * 72}"                        # une ligne de séparation
```

Le `:` sépare l'expression du format. `{r['accuracy']}` avec des apostrophes simples à
l'intérieur de guillemets doubles : obligatoire avant Python 3.12.

### 2.7 Le gestionnaire de contexte `with`

```python
with common.Stopwatch() as sw:
    train_result = trainer.train()
wall = sw.elapsed
```

`with` garantit qu'un traitement de sortie a lieu, même si le bloc lève une exception. La classe
`Stopwatch` implémente le protocole en deux méthodes :

```python
class Stopwatch:
    def __enter__(self):
        self.start = time.perf_counter()
        return self                     # ← ce que reçoit « as sw »
    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
        return False                    # ← False = ne pas avaler l'exception
```

Autre usage, dans `evaluate.py` : `with torch.inference_mode():` désactive le calcul des
gradients à l'intérieur du bloc et le réactive en sortant.

Et `with np.load(path) as data:` ferme le fichier même en cas d'erreur.

### 2.8 L'opérateur morse `:=`

Trois occurrences : deux dans `run_benchmark.py`, une dans `common.py`.

```python
if (code := train_script.main(flags)) != 0:          # run_benchmark.py (idem pour evaluate)
    return code

if (cached := sys.modules.get("_hf_evaluate")) is not None:   # common.load_hf_evaluate
    return cached
```

Il affecte **et** évalue en une expression. Sans lui il faudrait deux lignes. À lire : « appelle
`main`, range le résultat dans `code`, et si ce n'est pas 0, sors ».

### 2.9 `Path` plutôt que des chaînes

```python
ROOT = Path(__file__).resolve().parent
ckpt_dir = CHECKPOINTS_DIR / tag                # l'opérateur / assemble les chemins
if not (ckpt_dir / "config.json").exists():
path.write_text(contenu, encoding="utf-8")
for f in path.glob("*.safetensors"):
```

`pathlib.Path` gère les séparateurs de chemin à ta place : le même code marche sous Windows et
sous Linux. `__file__` est le chemin du fichier courant, `.resolve()` le rend absolu, `.parent`
remonte d'un cran — donc `ROOT` est toujours la racine du dépôt, quel que soit le dossier depuis
lequel on lance le script.

### 2.10 Le test `x == x` pour détecter un NaN

```python
if row["train_time_reduction_pct"] == row["train_time_reduction_pct"]:   # vrai sauf si NaN
```

Surprenant mais correct : par définition de la norme IEEE 754, `NaN != NaN`. C'est le seul
flottant qui n'est pas égal à lui-même. `math.isnan(x)` serait plus explicite ; l'idiome `x == x`
a l'avantage de fonctionner aussi sur des valeurs qui ne sont pas des flottants (`None` lèverait
une exception avec `isnan`).

Les NaN servent ici de « valeur absente » : si un modèle n'a pas de `train_*.json`, son temps
d'entraînement est NaN et les colonnes dérivées affichent `—`.

### 2.11 La convention du souligné `_`

`_features`, `_score_pass`, `_pt_to_data`, `_fmt` : le souligné initial signale « détail
d'implémentation, ne pas utiliser depuis un autre fichier ». Python ne l'empêche pas — c'est un
contrat entre développeurs. Une exception assumée : `selftest.py` appelle `plotting._colors_for`
et remplace `report.PREDICTIONS_DIR` (le notebook d'analyse, lui, réutilise `plotting._style_axes`),
parce que tester exige parfois d'entrer dans l'intimité du module.

### 2.12 `sys.exit(main())` et les codes de retour

```python
def main(argv: list[str] | None = None) -> int:
    ...
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

Deux idées :

- `if __name__ == "__main__":` n'est vrai que si le fichier est **lancé** (`python train.py`), pas
  s'il est **importé**. C'est ce qui permet à `run_benchmark.py` d'importer `train.py` et
  d'appeler `train_script.main([...])` sans que l'entraînement se déclenche à l'import.
- `main` renvoie un entier : `0` = succès, autre = échec. `sys.exit` le transmet au shell. C'est
  la convention Unix, et c'est ce qui permet `python report.py && echo ok`.
- `argv: list[str] | None = None` : si on ne passe rien, `argparse` lit `sys.argv`. Si on passe
  une liste, il lit cette liste — d'où la possibilité d'appeler `main(["--smoke", "--seed", "43"])`
  depuis un autre script.

### 2.13 Les imports tardifs

```python
def train_one(name, cfg, args):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
```

L'import est **dans** la fonction, pas en tête de fichier. Volontaire : `transformers` met
plusieurs secondes à s'importer. Grâce à ça, `python train.py --help` répond instantanément, et
`selftest.py` peut importer `report.py` et `plotting.py` sans avoir `transformers` installé.

---

## 3. Fichier par fichier

### 3.1 `config.py` — tout le déclaratif

Aucune logique, que des données. C'est le seul fichier à modifier pour changer le périmètre du
benchmark.

**Les chemins** (haut du fichier) : `RESULTS_DIR`, `PREDICTIONS_DIR`, `CHECKPOINTS_DIR`,
`PLOTS_DIR`, tous dérivés de `ROOT`. Juste en dessous, leur double pour `--smoke` :

```python
def smoke_dirs() -> dict[str, Path]:
    smoke = RESULTS_DIR / "smoke"
    return {"RESULTS_DIR": smoke, "PREDICTIONS_DIR": smoke / "predictions",
            "CHECKPOINTS_DIR": smoke / "checkpoints", "PLOTS_DIR": smoke / "plots"}
```

Les clés portent exactement les noms des variables globales. C'est ce qui permet, dans le `main`
de `train.py`, `evaluate.py` et `report.py`, ce geste un peu inhabituel :

```python
if args.smoke:
    globals().update(smoke_dirs())  # never overwrite a real run
```

`globals()` est le dictionnaire des variables globales **du module courant** ; le mettre à jour
re-lie `RESULTS_DIR` & co. pour tout le reste du script. Les fonctions qui lisent ces noms au
moment de l'appel voient donc les dossiers de fumée, sans qu'on ait à faire circuler un paramètre
`results_dir` dans chaque signature.

**`ModelSpec`** — une entrée du benchmark :

```python
@dataclass(frozen=True)
class ModelSpec:
    slug: str          # identifiant court, sert dans les noms de fichiers
    checkpoint: str    # l'id Hugging Face, ex. "bert-base-uncased"
    label: str         # nom d'affichage dans les tableaux et figures
    lr: float          # learning rate recommandé pour CE modèle
    family: str
    note: str
```

Le champ `lr` porte une décision méthodologique. Les petits modèles distillés supportent — et
demandent — un LR plus grand que BERT-base. Deux questions différentes s'offrent :

- *« quel est le meilleur atteignable par architecture ? »* → chaque modèle avec son LR (défaut)
- *« comment se comportent-ils à réglage constant ? »* → `--uniform-lr`

Ce sont deux expériences distinctes, avec deux réponses possiblement différentes. Le rapport
indique laquelle a tourné, dans « Method notes ».

**Le registre** :

```python
MODELS: dict[str, ModelSpec] = {
    "bert-base":  ModelSpec(slug="bert-base",  checkpoint="bert-base-uncased",  lr=3e-5, …),
    "distilbert": ModelSpec(slug="distilbert", checkpoint="distilbert-base-uncased", lr=5e-5, …),
    "minilm-l12": ModelSpec(slug="minilm-l12", checkpoint="microsoft/MiniLM-L12-H384-uncased", …),
    # ---- extended set: opt in with --models or --sweep (see SWEEPS below) ---- #
    "modernbert": …,
    "deberta-v3-xsmall": …,
    …                     # 17 entrées au total, jusqu'à "bert-tiny"
}
DEFAULT_MODELS = ["bert-base", "distilbert", "minilm-l12"]
SWEEPS = {
    "classic":   DEFAULT_MODELS,
    "modern":    ["bert-base", "distilbert", "minilm-l12", "ettin-32m", "ettin-68m"],
    "mechanisms": [...], "scaling": [...], "families": [...], "vocabulary": [...], …,
    "all":       list(MODELS),
}
BASELINE = "bert-base"      # toutes les quantités relatives sont calculées contre lui
```

Le registre compte **dix-sept** modèles : les trois du défaut, puis un jeu étendu qui commence
par ModernBERT-base et descend jusqu'à BERT-tiny (DeBERTa, ELECTRA, ALBERT, RoBERTa, les Ettin,
MiniLM-L6, BERT-mini…). On n'en lance jamais dix-sept par défaut : `SWEEPS` regroupe des
sous-ensembles nommés, que `run_benchmark.py --sweep NOM` sélectionne. Chaque sweep est **une
question**, pas une tranche de taille (« à taille égale, un modèle distillé de 2019 ou un petit
modèle de 2025 ? » pour `modern`, « quatre façons d'obtenir un modèle moins cher » pour
`mechanisms`…), et tous gardent `bert-base` pour que les colonnes relatives restent comparables
d'un sweep à l'autre.

Ajouter un modèle = ajouter une entrée. Tout le reste (tableaux, statistiques, figures,
couleurs) le prend en compte automatiquement. C'est le test qu'un découpage est bon : une
extension prévue doit se faire en un seul endroit.

**`DatasetSpec`** décrit où trouver le texte et le label :

```python
"ag_news": DatasetSpec(
    hf_name="ag_news", text_column="text", label_column="label",
    num_labels=4, train_split="train", test_split="test",
    label_names=["World", "Sports", "Business", "Sci/Tech"],
    recommended_max_length=128,
    note="… 128 tokens couvrent ~99% des exemples …",
)
```

`recommended_max_length` encode une vraie contrainte : AG News tient dans 128 tokens, IMDB non.
Sur IMDB à 256 tokens, tous les modèles voient le même préfixe tronqué — la colonne accuracy
mesure alors « accuracy à ce budget de tokens », pas le plafond des modèles. Le `note` est repris
tel quel dans le rapport pour que le lecteur le sache.

**`BenchConfig`** — les réglages, avec les valeurs par défaut dimensionnées pour un CPU de
portable :

```python
@dataclass
class BenchConfig:
    dataset: str = DEFAULT_DATASET          # "ag_news"
    max_length: int = 128
    max_train_samples: int | None = 4000
    max_eval_samples: int | None = 2000
    epochs: float = 1.0
    train_batch_size: int = 16
    eval_batch_size: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1               # part des pas d'entraînement en montée du LR
    seed: int = 42

    # --- inference-latency protocol ---
    latency_warmup: int = 10        # passes NON chronométrées : au moins ce nombre…
    latency_warmup_s: float = 1.0   # … ET au moins cette durée
    latency_samples: int = 200      # passes chronométrées à batch=1
    latency_rounds: int = 3         # tours entrelacés (voir §3.4)
    throughput_batches: int = 20    # batchs chronométrés un par un
    throughput_warmup_s: float = 3.0
    num_threads: int | None = None
```

Les commentaires du vrai fichier sont longs, et ils méritent d'être lus : ils racontent pourquoi
la chauffe a un **plancher de durée** en plus d'un nombre de passes (10 passes chauffent MiniLM
0,35 s et BERT-base 0,95 s — le petit modèle serait mesuré sur une machine plus froide), et
pourquoi `latency_rounds` existe (le même modèle mesuré 70, 92, 74 puis 70 ms sur quatre runs).

Et `SMOKE_OVERRIDES` : un dictionnaire de valeurs minuscules appliquées par `--smoke` (200
exemples, 20 passes chronométrées, un seul tour, chauffes raccourcies), pour prouver que le
pipeline tourne en quelques minutes sans attendre une demi-heure.

---

### 3.2 `common.py` — la boîte à outils

Dix sections, séparées par des bannières de commentaires : *Reproducibility*, *Registry
helpers*, *Data*, *Model accounting*, *Metrics*, *Statistics*, *Timing*, *Environment capture*,
*JSON I/O*, *transformers version compatibility*. Je prends les plus importantes.

#### Reproductibilité — `set_seed`

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch
    torch.manual_seed(seed)
```

Il y a quatre générateurs aléatoires distincts dans la pile (Python, NumPy, le hachage des
chaînes, PyTorch). En semer un seul ne suffit pas. Ce qui dépend de l'aléa ici : l'initialisation
de la tête de classification, l'ordre de mélange du dataset, le dropout.

#### Données — `load_splits`

```python
ds = load_dataset(spec.hf_name)
train = ds[spec.train_split]
if max_train_samples is not None and max_train_samples < len(train):
    train = train.shuffle(seed=seed).select(range(max_train_samples))
```

`shuffle(seed=…)` **puis** `select` : c'est l'ordre qui compte. Sans mélange, `select(range(4000))`
prendrait les 4000 premiers exemples — or AG News est trié par classe, on n'aurait qu'un ou deux
labels. Avec un seed fixe, les trois modèles reçoivent **exactement** les mêmes 4000 exemples.

C'est cette égalité qui autorise les tests appariés plus loin. Sans elle, un écart d'accuracy
mélangerait la différence entre modèles et la différence entre jeux d'évaluation.

#### Données — `tokenize_splits`

```python
def _tok(batch):
    return tokenizer(batch[spec.text_column], truncation=True,
                     padding="max_length", max_length=max_length)

keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}

def _prep(split):
    out = split.map(_tok, batched=True, desc="tokenizing")
    if spec.label_column != "labels":
        out = out.rename_column(spec.label_column, "labels")
    out = out.remove_columns([c for c in out.column_names if c not in keep])
    out.set_format("torch")
    return out
```

Quatre points :

- **`padding="max_length"`** et non `padding=True`. Le padding dynamique remplirait chaque batch
  jusqu'à son plus long élément, donc la géométrie des tenseurs varierait d'un batch à l'autre.
  En longueur fixe, tous les modèles traitent exactement la même forme — la latence compare des
  architectures, pas des hasards de découpage. Le prix : on calcule sur du padding inutile. C'est
  le compromis assumé d'un banc de mesure contrôlé.
- **`rename_column` vers `"labels"`** : `Trainer` cherche une colonne nommée exactement `labels`.
  Le renommage est gardé par un `if` : `rename_column` lève une erreur si la colonne source porte
  déjà le nom cible, ce qui arriverait avec un dataset dont le label s'appelle déjà `labels`.
- **`remove_columns`** : on jette la colonne `text` originale. Sinon `Trainer` essaierait de la
  convertir en tenseur et échouerait.
- **`token_type_ids` dans `keep`** : BERT en produit, DistilBERT non. Comme chaque modèle
  tokenise avec **son** tokenizer, la colonne est là ou pas, et le `keep` s'adapte sans code
  conditionnel.

#### Comptage — `count_parameters`

```python
total = sum(p.numel() for p in model.parameters())
embedding = 0
for name, module in model.named_modules():
    if module.__class__.__name__ == "Embedding":
        embedding += sum(p.numel() for p in module.parameters(recurse=False))
return {"params_total": total, "params_embedding": embedding,
        "params_non_embedding": total - embedding, …}
```

La séparation embeddings / reste est **le** détail qui change l'interprétation. BERT-base et
DistilBERT partagent des embeddings identiques d'environ 23,8 M paramètres (la table des tokens,
23,4 M, plus les embeddings de position). Donc :

- en paramètres totaux : DistilBERT coupe **38,8 %**
- en paramètres hors embeddings : DistilBERT coupe **49,7 %**

Or les embeddings sont une **table de correspondance** : un `input_id` sert d'indice, on lit une
ligne. Coût de calcul quasi nul. Le temps d'inférence est dominé par les matmuls de l'encodeur.
C'est donc la seconde figure qui explique la latence, et la première qui explique la taille du
fichier. Reporter les deux évite une confusion classique.

`module.parameters(recurse=False)` ne compte que les paramètres du module lui-même, sans ses
enfants — sinon on compterait deux fois.

#### Le contournement d'import — `load_hf_evaluate`

Un vrai piège, hérité du nom de fichier `evaluate.py` demandé par la structure du projet.

Le problème : quand Python lance `python train.py`, il met le dossier du script **en premier**
dans `sys.path`. Donc `import evaluate` trouve `./evaluate.py` — notre script — et pas la
bibliothèque Hugging Face du même nom. Sans traitement, `compute_metrics` échouerait
silencieusement et retomberait toujours sur scikit-learn.

La solution : demander au chercheur de modules d'ignorer notre dossier.

```python
if (cached := sys.modules.get("_hf_evaluate")) is not None:
    return cached                                                   # ← déjà chargé : on réutilise

here = Path(__file__).resolve().parent
search = [p for p in sys.path if p and Path(p).resolve() != here]   # ← on retire notre dossier
spec = importlib.machinery.PathFinder.find_spec("evaluate", path=search)
if spec is None or spec.loader is None:
    return None

module = importlib.util.module_from_spec(spec)
sys.modules["_hf_evaluate"] = module

previous = sys.modules.get("evaluate")
sys.modules["evaluate"] = module        # ← et on l'y LAISSE
try:
    spec.loader.exec_module(module)
except Exception as exc:
    if previous is not None:            # rollback seulement en cas d'échec
        sys.modules["evaluate"] = previous
    else:
        sys.modules.pop("evaluate", None)
    sys.modules.pop("_hf_evaluate", None)
    print(f"[metrics] could not load the `evaluate` library: {type(exc).__name__}: {exc}")
    return None

if not callable(getattr(module, "load", None)):
    sys.modules.pop("_hf_evaluate", None)   # quelque chose d'autre répondait au nom `evaluate`
    return None
return module
```

Pourquoi `sys.modules["evaluate"]` doit-il pointer vers la bibliothèque ? Parce que le package
Hugging Face résout ses propres sous-modules par import **absolu** à travers ce nom — pendant
l'exécution du module, mais aussi **plus tard**, paresseusement, au premier accès à un attribut
(les packages Hugging Face diffèrent le chargement de leurs sous-modules).

**L'histoire du bug** vaut d'être racontée, parce qu'elle est instructive. Une version précédente
faisait ce qui semble le plus propre : poser la liaison, exécuter le module, puis **restaurer**
l'état d'avant dans un `finally`, « pour ne rien laisser traîner ». Tout semblait marcher : le
module se chargeait, aucune erreur. Mais au premier `evaluate.load(...)`, le chargement paresseux
ré-importait `evaluate`, trouvait **notre** `evaluate.py` en tête de `sys.path`, et levait une
`AttributeError`. `compute_metrics` l'attrapait et retombait sur scikit-learn — **à chaque run,
silencieusement**. Les chiffres restaient justes (les deux chemins calculent la même chose), mais
l'affirmation « métriques calculées avec `evaluate` » était fausse, et rien ne le signalait.

D'où le code actuel :

- la liaison **persiste** après le retour de la fonction ; elle n'est annulée que si
  `exec_module` échoue (dans ce cas, l'état d'avant est intact) ;
- l'échec est **imprimé**, avec le type d'exception et le message, au lieu d'être avalé ;
- la vérification `callable(load)` refuse un module qui porterait le nom sans être la bibliothèque ;
- le cache `_hf_evaluate` évite de ré-exécuter le package à chaque appel (`compute_metrics` appelle
  la fonction pour chaque modèle).

La leçon générale : un nettoyage « par hygiène » n'est pas neutre. Restaurer un état global
suppose que plus personne n'en a besoin — ici c'était faux, et l'échec était invisible parce
qu'un repli silencieux le masquait. `selftest.py` garde désormais la régression
(`test_evaluate_shim` vérifie que `sys.modules['evaluate']` reste lié à la bibliothèque).

`run_benchmark.py` fait le geste miroir : il charge ses voisins **par chemin de fichier**, ce qui
ne dépend d'aucun ordre d'import.

```python
spec = importlib.util.spec_from_file_location(f"_bench_{name}", ROOT / f"{name}.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
```

#### Statistiques — c'est ici que le projet se distingue

**`wilson_interval`** — intervalle de confiance à 95 % sur une proportion.

```python
p = successes / n
denom  = 1 + z*z/n
center = (p + z*z/(2*n)) / denom
half   = (z/denom) * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
return (max(0.0, center - half), min(1.0, center + half))
```

Soit, avec $z = 1{,}96$ et $\hat p = k/n$ :

$$\text{centre} = \frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}, \qquad
\text{demi-largeur} = \frac{z}{1 + \frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}$$

Pourquoi pas l'approximation normale $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ ? Deux défauts
rédhibitoires ici : elle sort de $[0,1]$ quand $\hat p$ est proche de 1, et sa couverture réelle
s'effondre dans ce régime. Or on travaille précisément à $\hat p \approx 0{,}90$. Wilson inverse
le test du score au lieu de linéariser — l'intervalle reste dans $[0,1]$ et garde sa couverture
nominale (le `max`/`min` final ne fait que protéger des arrondis flottants). On note au passage que le centre de Wilson n'est pas $\hat p$ : il est tiré vers
$1/2$, d'autant plus que $n$ est petit.

**`mcnemar_exact`** — le test de comparaison de deux classifieurs sur un même jeu de test.

L'idée : on ne compare pas deux accuracy comme deux nombres indépendants, on regarde le tableau
de contingence des **paires**.

|  | B a bon | B a faux |
|---|---|---|
| **A a bon** | $a$ | $b$ |
| **A a faux** | $c$ | $d$ |

Les cases $a$ et $d$ (les deux d'accord) ne disent **rien** sur lequel est meilleur. Toute
l'information est dans les **discordants** $b$ et $c$. Sous l'hypothèse nulle « les deux modèles
sont équivalents », chaque discordance est un tirage à pile ou face : $b \sim
\mathrm{Bin}(b+c,\,1/2)$.

```python
b = int(np.sum(correct_a & ~correct_b))     # A bon, B faux
c = int(np.sum(~correct_a & correct_b))     # A faux, B bon
n = b + c
if n == 0:
    return {"p_value": 1.0, …}
k = min(b, c)
tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
p = min(1.0, 2.0 * tail)
```

C'est-à-dire

$$p = 2 \sum_{i=0}^{\min(b,c)} \binom{n}{i} \left(\tfrac12\right)^{n}, \qquad n = b+c,$$

tronqué à 1. Version **exacte** (binomiale), pas l'approximation du $\chi^2$ : avec quelques
dizaines de discordances seulement, l'approximation asymptotique n'est pas justifiée. Et aucune
dépendance à SciPy, `math.comb` suffit.

Vérification dans `selftest.py` : pour $b=10$, $c=1$, on attend
$p = 2\,(\binom{11}{0}+\binom{11}{1})/2^{11} = 24/2048 = 0{,}011719$. Le code renvoie exactement ça.

Les opérateurs `&` et `~` sont le **et** et le **non** bit à bit, qui s'appliquent élément par
élément sur des tableaux NumPy de booléens. `correct_a & ~correct_b` est un tableau de booléens,
et `np.sum` d'un tableau de booléens compte les `True`.

**`paired_bootstrap_delta`** — intervalle de confiance sur la **différence** d'accuracy.

```python
n = len(correct_a)
rng = np.random.default_rng(seed)
idx = rng.integers(0, n, size=(n_boot, n))                 # 10 000 × n indices tirés avec remise
deltas = correct_a[idx].mean(axis=1) - correct_b[idx].mean(axis=1)
lo, hi = np.percentile(deltas, [2.5, 97.5])
```

Trois lignes qui méritent un arrêt.

`idx` est une matrice $10\,000 \times n$ d'indices tirés uniformément avec remise. `correct_a[idx]`
est de l'**indexation avancée** NumPy : le résultat a la forme de `idx`, soit $10\,000 \times n$,
chaque ligne étant un ré-échantillonnage du vecteur de justesse. `.mean(axis=1)` moyenne le long
des colonnes, donnant 10 000 accuracy simulées. Tout se fait sans boucle Python.

Le point clé est **apparié** : le même `idx` sert pour les deux modèles. Un exemple tiré est tiré
pour A *et* pour B. On échantillonne donc la distribution de la différence, et non la différence
de deux distributions indépendantes. Résultat : l'intervalle est bien plus serré, parce que la
variabilité commune aux deux modèles (« cet exemple est difficile pour tout le monde ») se
soustrait au lieu de s'ajouter. C'est aussi pour ça que différencier deux intervalles de Wilson
serait faux — trop large, donc trop conservateur.

**`pareto_front`** — les points non dominés, pour (maximiser l'accuracy, minimiser la latence).

```python
for i, (acc_i, lat_i) in enumerate(points):
    dominated = any(
        (acc_j >= acc_i and lat_j <= lat_i) and (acc_j > acc_i or lat_j < lat_i)
        for j, (acc_j, lat_j) in enumerate(points) if j != i
    )
    if not dominated:
        front.append(i)
```

$j$ **domine** $i$ si $j$ est au moins aussi bon sur les deux critères et strictement meilleur
sur au moins un. La double condition est la définition standard de la dominance de Pareto ;
elle évite qu'un point se déclare dominé par un point identique. `any(...)` s'arrête au premier
dominateur trouvé.

#### Chronométrage

```python
class Stopwatch:            # voir §2.7
def summarize_latencies(samples_s):
    ms = [s * 1000.0 for s in samples_s]       # dans l'ordre de mesure
    ordered = sorted(ms)                        # copie triée, à part
    n = len(ordered)
    return {"latency_ms_p50": float(ordered[int(0.50 * (n - 1))]),
            "latency_ms_p95": float(ordered[int(0.95 * (n - 1))]), …,
            "latency_drift": drift_ratio(ms),   # ← lit la série NON triée
            "latency_n": n}
```

`time.perf_counter()` et non `time.time()` : compteur monotone de haute résolution, insensible
aux ajustements d'horloge système. Et on rapporte **médiane et p95**, pas la moyenne : une seule
pause du ramasse-miettes ou une préemption de l'OS déplace une moyenne, pas une médiane. Le p95
est le chiffre qui décide si un modèle tient dans un budget de requête — c'est la queue de
distribution qui fait rater les SLA, pas le cas moyen.

Les deux listes `ms` et `ordered` ne sont pas une coquetterie. `drift_ratio` compare la médiane du
dernier tiers des mesures à celle du premier tiers : il détecte une machine qui ralentit
**pendant** la mesure (sur CPU, la fréquence qui baisse sous charge vectorielle soutenue). Sur
une liste triée, le dernier tiers serait toujours plus lent que le premier, par construction —
l'indicateur ne voudrait plus rien dire. Les percentiles ont besoin de l'ordre trié, la dérive de
l'ordre chronologique ; on garde donc les deux.

La même section contient deux outils qui servent à `evaluate.py` (détaillés en §3.4) :
`calibration_probe`, qui chronomètre un produit matriciel fixe pour savoir à quelle vitesse tourne
la machine **en ce moment**, et `settle_machine`, qui charge le CPU jusqu'à ce que cette vitesse
cesse de changer.

#### `build_training_arguments` — la couche de compatibilité

```python
accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
if "eval_strategy" in kwargs and "eval_strategy" not in accepted:
    kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
dropped = sorted(set(kwargs) - accepted)
for key in dropped:
    kwargs.pop(key)
return TrainingArguments(**kwargs)
```

`transformers` a renommé `evaluation_strategy` en `eval_strategy` en version 4.46, et plusieurs
drapeaux sont apparus ou ont disparu au fil des versions. Plutôt que d'épingler une version
exacte, on **interroge la signature réelle** avec `inspect` et on s'y adapte. Le dépôt reste
donc utilisable sur une plage de versions au lieu d'exiger la bonne.

---

### 3.3 `train.py` — entraîner et chronométrer

Huit fonctions, dans l'ordre du fichier : `parse_args`, `build_config`, `resolve_lr`,
`planned_fingerprint`, `stored_fingerprint`, `stale_reasons`, `train_one`, `main`. Les trois
« empreintes » servent à une seule chose — décider si un checkpoint existant peut être réutilisé —
et se lisent avec `main`, à la fin.

**`parse_args`** déclare la ligne de commande. Un exemple qui montre le mécanisme :

```python
p.add_argument("--max-train-samples", type=int, default=4000, help="-1 for the full split")
p.add_argument("--uniform-lr", action="store_true")
p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS))
```

- `--max-train-samples` devient `args.max_train_samples` (les tirets deviennent des soulignés)
- `action="store_true"` : drapeau booléen, `False` sauf si présent
- `choices=list(MODELS)` : argparse refuse un nom inconnu, avec un message utile — la validation
  est gratuite parce que le registre est la source de vérité

**`build_config`** transforme les arguments en `BenchConfig` :

```python
cfg.max_length = args.max_length if args.max_length is not None else spec.recommended_max_length
cfg.max_train_samples = None if args.max_train_samples == -1 else args.max_train_samples
if args.smoke:
    for k, v in SMOKE_OVERRIDES.items():
        setattr(cfg, k, v)
```

Trois conventions : `None` en argument = « prends le défaut du dataset » ; `-1` = « prends tout
le split » ; `--smoke` écrase plusieurs champs d'un coup via `setattr`, qui affecte un attribut
dont le nom est dans une variable.

**`resolve_lr`** fixe la priorité entre les trois sources du learning rate :

```python
def resolve_lr(name, args):
    if args.lr is not None:
        return args.lr                                       # 1. --lr gagne toujours
    return MODELS[BASELINE].lr if args.uniform_lr else MODELS[name].lr   # 2. --uniform-lr, 3. défaut
```

C'est une fonction plutôt qu'une ligne dans `train_one` parce que `main` en a besoin **avant**
d'entraîner, pour calculer l'empreinte.

**`train_one`** — le cœur. Sa structure :

```python
lr = resolve_lr(name, args)

common.set_seed(cfg.seed)                      # 1. reproductibilité
threads = common.apply_thread_limit(cfg.num_threads)
device = common.resolve_device()

tokenizer = AutoTokenizer.from_pretrained(model_spec.checkpoint)      # 2. le modèle
model = AutoModelForSequenceClassification.from_pretrained(
    model_spec.checkpoint, num_labels=data_spec.num_labels, id2label=…)

params = common.count_parameters(model)                              # 3. l'inventaire
arch = common.architecture_summary(model)

train_raw, _ = common.load_splits(...)                               # 4. les données
trunc = common.truncation_rate(train_raw, tokenizer, data_spec, cfg.max_length)
train_ds, _ = common.tokenize_splits(...)

training_args = common.build_training_arguments(                     # 5. les réglages
    learning_rate=lr, weight_decay=cfg.weight_decay, warmup_ratio=cfg.warmup_ratio,
    eval_strategy="no", save_strategy="no", report_to=[], …)
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)

with common.Stopwatch() as sw:                                       # 6. LA MESURE
    train_result = trainer.train()
wall = sw.elapsed

model.save_pretrained(ckpt_dir)                                      # 7. on garde tout
tokenizer.save_pretrained(ckpt_dir)
common.write_json(RESULTS_DIR / f"train_{tag}.json", record)
```

Les décisions qui comptent :

- **`AutoModelForSequenceClassification`** ajoute une tête de classification linéaire sur
  l'encodeur pré-entraîné. Cette tête est initialisée au hasard — d'où l'avertissement de
  `transformers` au chargement, qui est normal ici. C'est justement ce qu'on entraîne.
- **`eval_strategy="no"`** : aucune évaluation pendant l'entraînement. Sinon le temps mesuré
  contiendrait des passes d'évaluation, et la colonne « train time » ne voudrait plus rien dire.
- **`save_strategy="no"`** puis `save_pretrained` à la main : on ne veut pas que `Trainer` écrive
  des checkpoints intermédiaires (lents, volumineux), juste l'état final.
- **`report_to=[]`** : coupe les intégrations Weights & Biases / TensorBoard, qui sinon
  s'activent si elles sont installées et polluent la mesure.
- Le `Stopwatch` **n'entoure que `trainer.train()`**. Le chargement du modèle, la tokenisation,
  l'écriture sur disque sont dehors. On mesure l'entraînement, rien d'autre.

**`main`** commence par `globals().update(smoke_dirs())` si `--smoke` est passé (voir §3.1),
puis boucle sur les modèles, avec un raccourci utile — mais gardé :

```python
if not args.overwrite and train_json.exists() \
        and (CHECKPOINTS_DIR / tag / "config.json").exists():
    reasons = stale_reasons(
        planned_fingerprint(cfg, resolve_lr(name, args)),
        stored_fingerprint(common.read_json(train_json)),
    )
    if not reasons:
        print(f"[skip] {name}: checkpoint matches this configuration "
              f"(--overwrite to retrain anyway)")
        continue
    print(f"[retrain] {name}: configuration changed — {'; '.join(reasons)}")

train_one(name, cfg, args)
```

Relancer après une interruption ne réentraîne pas ce qui est déjà fait. Sur un CPU où chaque
modèle coûte 10 minutes, ça compte. Le `\` en fin de ligne poursuit l'expression à la ligne
suivante.

Mais un checkpoint n'est réutilisé **que s'il répond à la même question**. L'empreinte
(`planned_fingerprint`) rassemble les réglages qui déterminent ce que *sont* les poids : dataset,
seed, `max_length`, `max_train_samples`, époques, taille de batch, learning rate.
`stored_fingerprint` relit les mêmes champs dans le `train_<tag>.json` posé à côté du
checkpoint, et `stale_reasons` liste ce qui diffère, sous la forme `"max_train_samples: 200 ->
4000"`. Liste vide = réutilisable.

Pourquoi ce garde-fou ? Parce que le raccourci a déjà menti. L'ancienne version sautait
l'entraînement dès que les fichiers existaient. Or un `--smoke` entraîne sur 200 exemples **sous
le même tag** qu'un vrai run : un vrai run lancé ensuite trouvait le checkpoint, sautait
l'entraînement, et rapportait sans un mot l'accuracy d'un modèle entraîné sur un vingtième des
données. (Depuis, `--smoke` écrit en plus sous `results/smoke/` ; les deux protections se
recouvrent volontairement.) Le même mécanisme rend les sweeps bon marché : `bert-base`, présent
dans tous, n'est entraîné qu'une fois tant que sa configuration ne change pas.

---

### 3.4 `evaluate.py` — mesurer accuracy et coût d'inférence

C'est le fichier le plus « expérimental » du dépôt : sa valeur tient au protocole. Commence par
la docstring du module, qui raconte le problème en dix lignes.

**L'idée directrice : trois phases.** L'accuracy est déterministe — les mêmes poids sur les mêmes
exemples donnent la même réponse, quel que soit le moment. Les temps, non : la vitesse d'un CPU
bouge avec sa fréquence, sa marge thermique, et tout ce que la machine fait d'autre. Mesurer
complètement le modèle A, puis complètement le modèle B, **confond** donc la dérive de la machine
avec l'identité du modèle : ce que faisait la machine pendant qu'on chronométrait A devient une
partie du chiffre de A. Mesuré ainsi sur un poste fluctuant, le même modèle a donné 70, 92, 74
puis 70 ms sur quatre runs, et l'accélération de DistilBERT variait de 1,6× à 2,5×. L'ordre
tenait ; les multiplicateurs étaient du bruit.

D'où la structure de `main` :

```python
backend = common.prefetch_metrics()                  # 0. le réseau AVANT toute mesure

prepared = {}                                        # 1. accuracy, une fois par modèle
for name in args.models:
    try:
        prepared[name] = score_one(name, cfg, device)
    except FileNotFoundError as exc:                 #    pas de checkpoint : on saute
        print(f"[skip] {exc}")
...                                                  #    (rien d'évalué → return 1)

settle = common.settle_machine()                     # 2. mise en régime globale…
common.write_json(RESULTS_DIR / "environment.json",
                  {**common.environment_info(), **settle, "latency_rounds": cfg.latency_rounds})
rounds = time_all(prepared, cfg, device)             #    … puis tours entrelacés

for name, item in prepared.items():                  # 3. agrégation et écriture
    timings = aggregate_timings(rounds[name])
    record = {**item["record"], **timings}
    common.write_json(RESULTS_DIR / f"eval_{item['tag']}.json", record)
```

On lit les phases une par une.

#### Avant tout : `prefetch_metrics`

Le premier `evaluate.load("accuracy")` va chercher un script sur le Hub : un aller-retour réseau
d'une dizaine de secondes. Placé à l'intérieur de l'évaluation du premier modèle, il laissait le
CPU inactif — donc à fréquence réduite — juste avant sa mesure : la sonde de calibration lisait
276 GFLOP/s contre ~550 pour tous les modèles suivants, un artefact ×2 qui se présentait comme
« le premier modèle est lent ». `prefetch_metrics` fait ce chargement une fois, en tête, et
renvoie le backend qui sera utilisé (`"evaluate"` ou `"sklearn"`). Leçon : toute entrée/sortie
doit sortir des régions mesurées, **et de ce qui les précède immédiatement**.

#### Phase 1 — `score_one` : l'accuracy

**`_features`** — une ligne, mais elle a une raison :

```python
def _features(batch):
    return {k: v for k, v in batch.items() if k != "labels"}
```

Si on passait `labels` au modèle, `transformers` calculerait la fonction de perte en plus des
logits. Ce serait du calcul en trop **dans la région chronométrée**. On enlève donc les labels —
partout, y compris dans la passe d'accuracy.

**`_score_pass`** — la passe qui produit les prédictions :

```python
with torch.inference_mode():
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        inputs = {k: v.to(device) for k, v in _features(batch).items()}
        logits = model(**inputs).logits
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        labels.append(batch["labels"].numpy())
return np.concatenate(preds), np.concatenate(labels), elapsed
```

- `range(0, len(dataset), batch_size)` : le troisième argument est le pas — on avance de 32 en 32.
- `dataset[i : i+batch_size]` : grâce à `set_format("torch")`, une tranche renvoie directement un
  dictionnaire de tenseurs empilés. Pas besoin de `DataLoader`, dont la surcharge polluerait
  la mesure.
- `logits.argmax(dim=-1)` : la classe prédite est l'indice du plus grand logit, sur la dernière
  dimension. On n'a pas besoin de passer par un softmax : il est monotone, donc il ne change pas
  l'argmax.
- `.cpu().numpy()` : on rapatrie sur CPU avant de convertir (obligatoire si on tournait sur GPU).
- `np.concatenate` recolle la liste de tableaux par batch en un seul vecteur.

**La suite de `score_one`** — ce qui est conservé :

```python
preds, labels, pass_s = _score_pass(model, test_ds, cfg.eval_batch_size, device)
metrics = common.compute_metrics(preds, labels)

correct = (preds == labels).astype(np.int8)
n_correct = int(correct.sum())
ci_low, ci_high = common.wilson_interval(n_correct, len(correct))
...
np.savez_compressed(PREDICTIONS_DIR / f"{tag}.npz",
                    preds=preds.astype(np.int16),
                    labels=labels.astype(np.int16),
                    correct=correct)

examples, batches = _timing_inputs(test_ds, cfg, device)
params = common.count_parameters(model)
record = {"slug": …, "n_correct": n_correct, **metrics,
          "accuracy_ci95_low": ci_low, "accuracy_ci95_high": ci_high,
          "full_pass_s": pass_s, **params, "size_on_disk_mb": …, …}
del model  # reloaded per round in phase 2, so nothing large is held here
return {"name": name, "tag": tag, "ckpt_dir": ckpt_dir, "record": record,
        "examples": examples, "batches": batches}
```

`preds == labels` produit un tableau de booléens exemple par exemple. **C'est ce vecteur qui rend
les tests appariés possibles** : sans lui on n'aurait que deux accuracy scalaires, et aucun moyen
de savoir *quels* exemples les distinguent. `.npz` est le format d'archive compressée de NumPy ;
`int16` au lieu de `int64` divise la taille par quatre pour des valeurs qui ne dépassent pas 4.

`score_one` ne chronomètre rien. Elle renvoie un **dossier préparé** pour la phase 2 : le record
partiel, le chemin du checkpoint, et les tenseurs déjà construits par `_timing_inputs` :

```python
n = min(cfg.latency_samples, len(test_ds))
examples = [{k: v.unsqueeze(0).to(device) for k, v in _features(test_ds[i]).items()}
            for i in range(n)]

bs = cfg.eval_batch_size
n_batches = min(cfg.throughput_batches, max(1, len(test_ds) // bs))
batches = [{k: v.to(device) for k, v in _features(test_ds[i * bs : (i + 1) * bs]).items()}
           for i in range(n_batches)]
```

- **`unsqueeze(0)`** ajoute une dimension de batch : un exemple de forme `(128,)` devient
  `(1, 128)`. Le modèle attend toujours un batch, même de taille 1.
- **Les tenseurs sont préparés une fois, avant toute mesure**, et réutilisés à chaque tour. Sinon
  on chronométrerait aussi la conversion et le transfert vers le device. Quelques mégaoctets par
  modèle : le prix est négligeable.
- `//` est la division entière : `len(test_ds) // bs` = nombre de batchs complets disponibles.

Le modèle, lui, est jeté (`del model`) : on le rechargera à chaque tour.

#### Phase 2 — `time_all` : les tours entrelacés

```python
for r in range(cfg.latency_rounds):
    order = _rotated(names, r)
    for name in order:
        item = prepared[name]
        model = _load_model(item["ckpt_dir"], device)
        calib = common.settle_machine(min_seconds=1.0, max_seconds=8.0,
                                      compare_seconds=0.5, verbose=False)
        latency = _measure_latency(model, item["examples"], cfg)
        throughput = _measure_throughput(model, item["batches"], cfg)
        rounds[name].append({**latency, **throughput,
                             "calibration_gflops": calib["settled_gflops"],
                             "calibration_converged": bool(calib["settle_converged"])})
        del model
```

Chaque tour mesure **tous** les modèles. Si la machine ralentit au milieu du deuxième tour, ce
ralentissement touche un peu tout le monde au lieu de s'attacher entièrement à un seul modèle ;
et l'écart d'un tour à l'autre devient une **mesure** de l'incertitude au lieu d'un biais caché.

**`_rotated`** fait tourner l'ordre de départ :

```python
def _rotated(names, offset):
    k = offset % len(names)
    return names[k:] + names[:k]        # tour 0 : A B C ; tour 1 : B C A ; tour 2 : C A B
```

À l'intérieur d'un tour, les modèles passent toujours l'un après l'autre, donc la position porte
encore une petite pénalité. La rotation la répartit au lieu de la concentrer : avec autant de
tours que de modèles, chacun occupe chaque position exactement une fois. `%` (modulo) fait
reboucler l'indice quand il y a plus de tours que de modèles.

**Le modèle est rechargé à chaque tour**, puis `settle_machine` tourne **avant** de mesurer. Le
rechargement fait des lectures disque, qui laissent le CPU inactif et donc à fréquence réduite ;
sans remise en régime, la sonde mesurerait le creux plutôt que la machine. Le `calib` obtenu est
rangé à côté des temps : c'est le témoin indépendant du modèle.

**`_warmup`** — la chauffe, avec un double plancher :

```python
def _warmup(model, inputs, min_iters, min_seconds):
    start = time.perf_counter()
    done = 0
    while done < min_iters or (time.perf_counter() - start) < min_seconds:
        model(**inputs[done % len(inputs)])
        done += 1
        if done >= 10_000:  # pathological guard
            break
    return done
```

Les premières passes sont anormalement lentes : allocation des tampons, initialisation paresseuse
des noyaux, caches froids. Il faut donc les jeter. Mais un **nombre** fixe de passes ne suffit
pas : 10 passes chauffent un modèle rapide bien moins longtemps qu'un lent (0,35 s pour MiniLM,
0,95 s pour BERT-base), et le petit modèle serait mesuré sur une machine plus froide. La boucle
continue donc tant que **l'une ou l'autre** condition n'est pas remplie : au moins `min_iters`
passes **et** au moins `min_seconds` secondes. Le nombre de passes effectives est renvoyé et
enregistré (`latency_warmup_iters`).

**`_measure_latency`** — le protocole à batch = 1 :

```python
with torch.inference_mode():
    warmed = _warmup(model, examples, cfg.latency_warmup, cfg.latency_warmup_s)
    samples = []
    for ex in examples:
        t0 = time.perf_counter()
        model(**ex)
        samples.append(time.perf_counter() - t0)
return {**common.summarize_latencies(samples), "latency_warmup_iters": warmed}
```

Chaque passe est chronométrée individuellement, et `summarize_latencies` (§3.2) en tire médiane,
p95 et dérive. **`torch.inference_mode()`** désactive le suivi des gradients ; plus fort que
`no_grad()`, il supprime aussi le compteur de version des tenseurs.

**`_measure_throughput`** répond à une **autre** question :

```python
with torch.inference_mode():
    warmed = _warmup(model, batches, min(cfg.latency_warmup, len(batches)),
                     cfg.throughput_warmup_s)
    per_batch = []
    for b in batches:
        t0 = time.perf_counter()
        model(**b)
        per_batch.append(time.perf_counter() - t0)

median_s = statistics.median(per_batch)
bs = cfg.eval_batch_size
return {"throughput_examples_per_s": bs / median_s,
        "throughput_drift": common.drift_ratio(per_batch), …}
```

Chaque batch est chronométré **séparément**, et le débit vaut taille de batch / temps médian d'un
batch. Un seul chronomètre autour de la boucle donnerait une moyenne, sensible à une pause
isolée, et cacherait une machine qui ralentit en cours de route ; ici la médiane est robuste et
`drift_ratio` rend la dérive visible.

Latence et débit ne sont pas la même chose et ne se déduisent pas l'un de l'autre. À batch = 1 le
processeur est sous-utilisé et le coût par exemple est dominé par les surcoûts fixes. À batch = 32
les matmuls saturent les unités de calcul et le coût par exemple s'effondre. La latence décide si
un modèle tient dans une requête interactive ; le débit décide combien coûte le scoring d'un
corpus entier. Le rapport donne les deux.

**`settle_machine`, version globale.** Avant le premier tour, `main` appelle `settle_machine()`
avec ses réglages par défaut (jusqu'à 45 s). Un processeur au repos n'est pas à sa fréquence de
croisière et met plusieurs secondes de charge à l'atteindre : sans cette étape, le premier modèle
mesuré rencontrait une machine plus lente que tous les suivants (227 GFLOP/s contre 480–620, un
écart ×2,7). Aucune chauffe par modèle ne peut corriger ça : le biais vit **entre** les modèles.
La convergence compare deux **fenêtres de temps** successives (la dernière seconde et demie
contre la précédente), et non des sondes consécutives : trois sondes à la suite couvrent 200 ms,
trop peu pour voir une montée progressive. Le résultat (convergé ou non, vitesse de départ et
d'arrivée) est écrit dans `results/environment.json`, avec les versions et le nombre de tours.

#### Phase 3 — `aggregate_timings` : la médiane et l'étendue

```python
def med(key):    return float(statistics.median([s[key] for s in samples]))
def spread(key):
    values = [s[key] for s in samples]
    return float(max(values) / min(values)) if min(values) > 0 else float("nan")

return {"latency_rounds": len(samples),
        "latency_ms_p50": med("latency_ms_p50"),
        "latency_ms_p50_min": float(min(p50s)), "latency_ms_p50_max": float(max(p50s)),
        "latency_p50_spread": spread("latency_ms_p50"),
        …,
        "calibration_gflops": med("calibration_gflops"),
        "calibration_spread": spread("calibration_gflops"),
        "calibration_converged": all(s["calibration_converged"] for s in samples)}
```

Pour chaque modèle, la valeur rapportée est la **médiane sur les tours** ; l'**étendue** (min, max,
et leur rapport `spread`) est ce qui la rend honnête. Un modèle dont le p50 a varié de 20 à 26 ms
d'un tour à l'autre n'a pas « une latence de 20 ms », et un nombre seul le cacherait. Le rapport
affiche cette étendue dans une colonne dédiée, et avertit en tête de document quand un modèle
dépasse 25 % d'écart (`report.timing_validity`).

`--latency-rounds 1` désactive l'entrelacement : plus rapide, mais la dérive redevient un biais.
C'est le réglage de `--smoke`.

---

### 3.5 `report.py` — l'analyse

Le fichier le plus long, mais très linéaire : charger, calculer, formater.

**`load_rows`** — la jointure :

```python
for name, spec in MODELS.items():
    tag = common.run_tag(spec.slug, dataset, seed)
    if not (RESULTS_DIR / f"eval_{tag}.json").exists():
        continue                                   # ← ce modèle n'a pas tourné, on l'ignore
    row = common.read_json(RESULTS_DIR / f"eval_{tag}.json")
    if (RESULTS_DIR / f"train_{tag}.json").exists():
        train = common.read_json(...)
        for key in ("train_time_s", "train_samples_per_s", …):
            row[key] = train.get(key)
    row.setdefault("train_time_s", float("nan"))   # ← pas d'entraînement : NaN
    row["tag"] = tag
    rows.append(row)

order = list(MODELS)
rows.sort(key=lambda r: order.index(r["slug"]) if r["slug"] in order else 99)
```

Un `row` est un dictionnaire plat qui fusionne l'enregistrement d'évaluation et celui
d'entraînement. Le tri se fait selon l'ordre du registre `MODELS` dans `config.py` — l'ordre
d'insertion du dictionnaire, pas un ordre de taille : la baseline d'abord, puis les modèles du
défaut, puis le jeu étendu (Ettin-68M, plus gros que MiniLM, apparaît donc après lui). `key=lambda …`
donne à `sort` la valeur sur laquelle comparer.

Le fichier tolère l'absence : évaluer sans avoir entraîné donne des colonnes de temps à NaN, mais
le rapport se construit quand même.

**`analyse`** — toutes les quantités dérivées. Elles se rangent en trois familles.

*Les rapports*, sans dimension, qui se transportent d'une machine à l'autre :

```python
row["accuracy_retention"]   = row["accuracy"] / base["accuracy"]
row["delta_accuracy_pp"]    = 100.0 * (row["accuracy"] - base["accuracy"])
row["latency_speedup"]      = base[LATENCY_KEY] / row[LATENCY_KEY]
row["latency_reduction_pct"]= 100.0 * (1.0 - row[LATENCY_KEY] / base[LATENCY_KEY])
row["param_reduction_pct"]  = 100.0 * (1.0 - row["params_total"] / base["params_total"])
```

Attention à l'asymétrie : accélération $\times$ et réduction en % sont deux écritures du même
fait, liées par $\text{réduction} = 1 - 1/\text{accélération}$. $2\times$ plus rapide, c'est
$50\,\%$ de moins ; $3{,}09\times$, c'est $67{,}6\,\%$. Les deux sont dans le rapport parce que
les lecteurs ne pensent pas tous dans la même unité.

*Les tests appariés* :

```python
row_correct = load_correctness(row["tag"])
if base_correct is not None and row_correct is not None and not row["is_baseline"]:
    if len(base_correct) != len(row_correct):
        print("[warn] eval sets differ in size; skipping paired tests")
    else:
        boot = common.paired_bootstrap_delta(row_correct, base_correct, seed=seed)
        mcnemar = common.mcnemar_exact(base_correct, row_correct)
        row["significant_at_05"] = bool(mcnemar["p_value"] < 0.05)
```

Noter la garde sur les longueurs. Si deux runs ont été faits avec des `--max-eval-samples`
différents, l'appariement n'a pas de sens — le code refuse de calculer plutôt que de produire un
nombre faux. Refuser de répondre est parfois le bon comportement.

*Le taux de change* — la contribution la plus « ingénieur » du dépôt :

```python
loss_pp = max(0.0, -row["delta_accuracy_pp"])
extra_speed = row["latency_speedup"] - 1.0
row["accuracy_pp_per_extra_speedup"] = loss_pp / extra_speed if extra_speed > 0.01 else float("nan")
```

$$\varepsilon = \frac{\max(0,\ \text{acc}_{\text{base}} - \text{acc}_{\text{modèle}})\times 100}
{\text{accélération} - 1}$$

Combien de points d'accuracy pour chaque $\times$ de vitesse **supplémentaire** ? Le $-1$ au
dénominateur est essentiel : à accélération $1\times$ on n'a rien gagné, le dénominateur doit
s'annuler. Le garde-fou `> 0.01` évite la division par zéro pour un modèle qui n'accélère rien.
Le `max(0, …)` traite le cas où le petit modèle est *meilleur* : le taux de change vaut alors 0,
« gratuit », plutôt qu'un nombre négatif difficile à lire.

Une réserve, que le rapport écrit noir sur blanc : le numérateur est une **estimation ponctuelle**
de la perte d'accuracy. Quand cette perte n'est pas significative, le taux ne l'est pas non plus.
`build_markdown` ajoute alors une phrase à la ligne « Exchange rate » :

```python
noise = (
    " The accuracy loss behind it is not statistically detectable, so neither "
    "is this rate: it is not evidence that this model is a worse deal than one "
    "with a lower rate."
    if row.get("significant_at_05") is False and row["delta_accuracy_pp"] < 0 else ""
)
```

Dans les résultats committés, Ettin-68M affiche 0,641 pp par × et MiniLM-L12 0,623 : classer
l'un devant l'autre sur ce chiffre serait faux, puisque la perte d'Ettin-68M (p = 0,594) est dans
le bruit et celle de MiniLM (p = 0,005) ne l'est pas.

*La frontière de Pareto, et ce qui la domine.* Après les boucles par ligne, `analyse` calcule la
frontière avec `common.pareto_front` sur les couples (accuracy, latence), puis va plus loin. La
frontière est tracée sur des **estimations ponctuelles** : les écarts de latence sont grands et
stables, les écarts d'accuracy souvent non. Dire qu'un modèle est « battu sur les deux axes »
exige donc un test côté accuracy :

```python
correctness = {r["slug"]: load_correctness(r["tag"]) for r in rows}
for row in rows:
    row["dominated_by"] = []
    if row["on_pareto_frontier"]:
        continue
    for other in sorted(rows, key=lambda r: r[LATENCY_KEY]):
        if other is row or not (other["accuracy"] >= row["accuracy"]
                                and other[LATENCY_KEY] <= row[LATENCY_KEY]):
            continue
        a, b = correctness[other["slug"]], correctness[row["slug"]]
        p = (common.mcnemar_exact(a, b)["p_value"]
             if a is not None and b is not None and len(a) == len(b) else None)
        row["dominated_by"].append({"slug": …, "label": …, "accuracy_mcnemar_p": p})
```

Pour chaque modèle hors frontière, on liste **chaque** modèle qui le domine, avec le p de McNemar
entre les deux (pas contre la baseline : entre le dominé et son dominateur). `dominance_lines`
traduit ensuite en une puce par modèle : « *beaten on both axes by …* » si au moins un dominateur
a un p < 0,05, sinon « *beaten on latency; its accuracy deficit is not statistically
detectable* ». Même règle que partout ailleurs dans le rapport : un classement qui tient dans le
bruit n'est pas présenté comme un fait.

**`recommend`** — la décision, avec sa justification :

```python
candidates = [r for r in rows
              if not r["is_baseline"] and r.get("significant_at_05") is False
              and r["latency_speedup"] > 1.0]
if candidates:
    pick = max(candidates, key=lambda r: r["latency_speedup"])
    return {…, "basis": "fastest model with no statistically detectable accuracy gap (α = 0.05)"}
```

`r.get("significant_at_05") is False` et non `not r.get(...)`. La distinction est réelle :
`.get` renvoie `None` si la clé est absente (test non calculé), et `not None` vaut `True` — un
modèle non testé serait donc recommandé à tort. `is False` exige explicitement que le test ait
tourné et n'ait rien trouvé.

Le champ `basis` est renvoyé avec la recommandation et imprimé dans le rapport, pour qu'on sache
sur quel critère elle a été prise.

**`md_table`** — fabrique un tableau markdown aligné :

```python
widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
```

Pour chaque colonne, la largeur est le maximum entre l'en-tête et toutes les cellules. Le `*`
déballe le générateur en arguments de `max`. Puis chaque cellule est complétée par des espaces à
gauche ou à droite selon l'alignement voulu. C'est fait à la main plutôt qu'avec `tabulate` pour
ne pas ajouter une dépendance pour trente lignes de code.

**`build_markdown`** — l'assemblage, et un piège markdown :

```python
blocks: list[str] = [
    "# Benchmark results",
    f"Task: **{data_spec.hf_name}** …",
    f"Hardware: **{device}** …",
]
...
return "\n\n".join(blocks) + "\n"
```

Chaque élément de `blocks` est **un bloc markdown complet**, et on les joint par une ligne vide.
La première version concaténait des chaînes finissant par un seul `\n` : en markdown, deux lignes
consécutives séparées par un seul retour forment **un même paragraphe**. Les trois lignes
d'en-tête se retrouvaient collées. Joindre avec `"\n\n"` rend l'erreur structurellement
impossible.

Juste après l'en-tête vient `*timing_validity(rows)` : la liste des avertissements sur la
fiabilité des temps (étendue entre tours trop large, ou — pour un run à un seul tour — sondes de
calibration trop différentes d'un modèle à l'autre). Le `*` déballe la liste dans `blocks`. Les
réserves sont placées **avant** les tableaux : le lecteur doit rencontrer la mise en garde avant
le chiffre, pas après.

Enfin, `report.py` accepte lui aussi `--smoke` : il relit alors `results/smoke/` et écrit ses
figures sous `results/smoke/plots/`, par le même `globals().update(smoke_dirs())` que les deux
autres scripts.

---

### 3.6 `plotting.py` — les figures

Utile même si tu ne comptes pas modifier les graphiques, parce qu'il contient deux morceaux de
géométrie instructifs.

**Les thèmes** — deux dictionnaires de couleurs, clair et sombre :

```python
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "grid": "#e1e0d9",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"], …},
    "dark":  {"surface": "#1a1a19", "ink": "#ffffff", "grid": "#2c2c2a",
              "series": ["#3987e5", "#d95926", "#199e70"], …},
}
```

Les couleurs sombres ne sont pas l'inverse des claires : ce sont les mêmes trois teintes,
re-échelonnées pour un fond sombre. La palette a été validée pour la vision des couleurs — pire
paire, en distance perceptuelle sous deutéranopie, $\Delta E = 9{,}2$ en clair et $9{,}4$ en
sombre, pour un seuil de 8. Au-delà de trois modèles, `series_colors` retombe sur une seule
teinte plutôt que d'inventer une quatrième indistinguable :

```python
def series_colors(n, theme):
    slots = theme["series"]
    return list(slots[:n]) if n <= len(slots) else [slots[0]] * n
```

**`color_map`** — corrige un bug réel qu'avait la première version :

```python
def color_map(rows, theme):
    return {row["slug"]: c for row, c in zip(rows, series_colors(len(rows), theme))}
```

La figure « retention vs speedup » exclut la baseline (ses barres vaudraient 100 % et 1× par
construction). Si chaque figure calculait ses couleurs depuis **sa** liste, DistilBERT serait
orange dans une figure et bleu dans l'autre. La couleur doit suivre l'entité, jamais son rang
dans une liste filtrée. D'où une table `slug → couleur` construite une fois sur la liste
**complète** et passée à toutes les figures.

**La conversion d'unités** — le morceau à comprendre :

```python
def _pt_to_data(ax, points, axis):
    px = points * ax.figure.dpi / 72.0          # points typographiques → pixels
    inv = ax.transData.inverted()               # pixels → unités de données
    origin = inv.transform((0.0, 0.0))
    if axis == "y":
        return abs(inv.transform((0.0, px))[1] - origin[1])
    return abs(inv.transform((px, 0.0))[0] - origin[0])
```

Un point typographique vaut $1/72$ de pouce, donc `points × dpi / 72` donne des pixels.
`ax.transData` est la transformation données → pixels ; `.inverted()` fait l'inverse. On
transforme deux points et on prend la différence, ce qui donne la longueur en unités de données
correspondant à `points`.

**Le paramètre `axis` est indispensable**, et c'est là qu'était le bug. Prends le panneau de
latence des résultats committés (cinq modèles, p50 maximal 92,6 ms) : l'axe des ordonnées va de 0
à environ 111 ms (`max(values) * 1.20`) et celui des abscisses de $-0{,}6$ à $4{,}6$ (positions des
cinq catégories). Les deux axes ont des échelles qui diffèrent d'un facteur de l'ordre de vingt à
trente. Un même rayon de 3 pt correspond donc à un nombre d'unités de données **très** différent
selon l'axe — de l'ordre d'une à deux millisecondes en $y$, de quelques centièmes de catégorie en
$x$.

La première version calculait le rayon une seule fois, sur l'axe des valeurs, et l'appliquait
aussi à l'axe transverse. Le décalage horizontal du coin arrondi valait alors une ou deux unités
de catégorie au lieu de quelques centièmes — plus que l'espacement entre deux barres : d'où les
longs traits horizontaux en travers de chaque panneau, visibles sur la première version des
figures. Le correctif convertit le **même rayon en
pixels** séparément vers chaque axe :

```python
r_pt = max(0.0, min(BAR_ROUND_PT, width_pt / 2.0, height_pt))
path = _rounded_bar_path(..., _pt_to_data(ax, r_pt, cross_axis),
                              _pt_to_data(ax, r_pt, value_axis), horizontal)
```

Le `min` triple garantit aussi que le coin arrondi ne dépasse jamais la moitié de la largeur de
la barre ni sa hauteur — sinon une barre très courte deviendrait une forme absurde.

**Le chemin de la barre arrondie** — barre à extrémité arrondie, base carrée :

```python
codes = [MPath.MOVETO, MPath.LINETO,
         MPath.CURVE3, MPath.CURVE3,      # ← un coin = point de contrôle + point d'arrivée
         MPath.LINETO,
         MPath.CURVE3, MPath.CURVE3,      # ← l'autre coin
         MPath.LINETO, MPath.CLOSEPOLY]
verts = [(lo, base), (lo, inner),
         (lo, tip), (lo + r_cross, tip),
         (hi - r_cross, tip),
         (hi, tip), (hi, inner),
         (hi, base), (lo, base)]
```

Un `Path` matplotlib est une liste de sommets plus une liste de codes qui dit quoi faire de
chacun. `CURVE3` est une courbe de Bézier **quadratique** : elle se déclare par paires, le
premier sommet étant le point de contrôle et le second le point d'arrivée. Ici le point de
contrôle est le coin exact du rectangle, ce qui donne un arrondi régulier.

Pourquoi arrondir seulement le haut ? La base est ancrée sur le zéro de l'axe : c'est une ligne
de référence, elle doit être franche. L'extrémité porte la valeur : l'arrondi l'adoucit sans
déplacer le sommet.

**`fig.canvas.draw()` avant de tracer les barres** — subtilité importante :

```python
for ax, values, … in bar_panels:
    ax.set_xlim(...); ax.set_ylim(...)      # 1. on fixe les limites
    _titles(...); _style_axes(...)

fig.canvas.draw()                            # 2. les transformations sont calculées

for ax, values, … in bar_panels:
    rounded_bars(ax, positions, values, colors, theme)   # 3. maintenant on peut convertir
```

`_pt_to_data` a besoin de `ax.transData`, qui n'est correct qu'une fois la figure disposée et les
limites d'axes fixées. D'où la structure en deux boucles séparées par un `draw()` : c'est le prix
à payer pour convertir des pixels en unités de données.

**Le panneau accuracy en points, pas en barres.** Les écarts valent quelques dixièmes de point ;
il faut donc zoomer l'axe. Or une barre sur un axe tronqué ne code plus la grandeur par sa
longueur — c'est un procédé trompeur classique. On passe donc à des marqueurs avec moustaches
d'intervalle de confiance, où l'axe zoomé est parfaitement légitime. Et lire le **recouvrement**
des moustaches est justement l'information utile : là où elles se recouvrent, le classement de la
colonne n'est pas une preuve d'écart.

**`render`** — chaque figure est construite deux fois :

```python
def render(builder, out_dir, stem):
    for mode, suffix in (("light", ""), ("dark", "-dark")):
        theme = THEMES[mode]
        apply_theme(theme)
        fig = builder(theme)
        fig.savefig(out_dir / f"{stem}{suffix}.png", facecolor=theme["surface"], …)
        plt.close(fig)
```

`builder` est une **fonction passée en argument** (`lambda th: plotting.fig_pareto(rows, frontier,
th, cmap=plotting.color_map(rows, th), …)`),
appelée une fois par thème. Le README sert ensuite la bonne version via une balise `<picture>` et
`prefers-color-scheme`. `plt.close(fig)` libère la mémoire — sans lui, matplotlib accumule les
figures et prévient au bout de vingt.

---

### 3.7 `run_benchmark.py` — le chef d'orchestre

Court, et sans logique propre : il traduit ses arguments en listes de drapeaux et appelle les
`main` des trois scripts.

```python
def _shared(args, seed):
    flags = ["--models", *args.models, "--dataset", args.dataset, "--seed", str(seed)]
    if args.max_length is not None:
        flags += ["--max-length", str(args.max_length)]
    if args.num_threads is not None:
        flags += ["--num-threads", str(args.num_threads)]
    if args.smoke:
        flags.append("--smoke")
    return flags
```

`*args.models` déballe la liste dans la liste en construction : `["--models", "bert-base",
"distilbert", …]`. Tous les éléments sont des **chaînes**, parce que c'est ce que `argparse`
attend — d'où les `str(seed)`. `_shared` contient ce que **les deux** scripts doivent recevoir
à l'identique ; chacun reçoit ensuite ses drapeaux propres.

D'où vient `args.models` ? De `resolve_models` : `--models` (une liste ad hoc) ou `--sweep NOM`
(une liste nommée de `config.SWEEPS`), jamais les deux — passer les deux est une contradiction, et
le script la refuse plutôt que de trancher en silence. Sans l'un ni l'autre : les trois modèles
du défaut.

```python
for seed in args.seeds:
    if not args.skip_train:
        flags = _shared(args, seed) + ["--max-train-samples", …, "--epochs", …, "--batch-size", …]
        ...                                   # + --lr, --uniform-lr, --overwrite si demandés
        if (code := train_script.main(flags)) != 0:
            return code

    flags = _shared(args, seed) + [
        "--max-eval-samples", str(args.max_eval_samples),
        "--batch-size", str(args.eval_batch_size),
        "--latency-rounds", str(args.latency_rounds),
        "--latency-samples", str(args.latency_samples),
    ]
    if (code := evaluate_script.main(flags)) != 0:
        return code

report_flags = ["--dataset", args.dataset, "--seed", str(args.seeds[0]), "--baseline", args.baseline]
...                                           # + --seeds s'il y en a plusieurs
if args.smoke:
    report_flags.append("--smoke")
code = report_script.main(report_flags)
```

Boucle externe sur les seeds, propagation immédiate du premier code d'erreur non nul. Le rapport
n'est construit qu'une fois, à la fin, sur la première seed — les autres alimentent la section de
variance. `--smoke` est transmis aux trois étapes, y compris au rapport : sinon `report.py`
chercherait les résultats de fumée dans `results/` et ne les trouverait pas.

---

### 3.8 `selftest.py` — le filet de sécurité

Le fichier le plus utile au quotidien, parce qu'il tourne en quelques secondes sans
`transformers`. Six groupes de tests, appelés dans l'ordre par `main` : `test_statistics`,
`test_machine_probe` (la sonde et `settle_machine`, y compris une machine simulée qui monte en
vitesse et ne doit **pas** être déclarée stable), `test_evaluate_shim` (la régression de §3.2),
`test_checkpoint_reuse` (un checkpoint `--smoke` ne doit jamais être réutilisé pour un vrai run),
`test_analysis`, `test_figures`.

**Le mécanisme de vérification** :

```python
FAILURES: list[str] = []

def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {name}{'  — ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(name)
```

Volontairement sans framework de test : pas de `pytest` à installer, et la sortie se lit comme
une liste de propriétés vérifiées. Le programme continue après un échec pour tout signaler d'un
coup, et `main` renvoie 1 s'il en reste.

**Les assertions qui ont de la valeur** sont celles comparées à une vérité indépendante :

```python
lo, hi = common.wilson_interval(90, 100)
check("Wilson(90/100) matches the published interval",
      close(lo, 0.8256, 5e-4) and close(hi, 0.9448, 5e-4))

expected = 2 * (math.comb(11, 0) + math.comb(11, 1)) / 2**11
check("McNemar exact matches the hand-computed binomial tail",
      close(result["p_value"], expected, 1e-12))
```

Un test qui compare le code à lui-même ne prouve rien. Ceux-ci le comparent à une valeur publiée
et à un calcul refait à la main. Il y a aussi des tests de **propriété** — Wilson reste dans
$[0,1]$, l'intervalle se resserre quand $n$ croît, McNemar est symétrique en ses arguments — qui
attrapent des classes d'erreurs entières plutôt qu'un cas particulier.

**Les données synthétiques** : `synthetic_rows()` fabrique trois enregistrements de la forme
exacte que produit `evaluate.py`. Les valeurs sont inventées — elles servent à exercer le code,
pas à mesurer quoi que ce soit. `_fake_correctness` construit des vecteurs de justesse
**emboîtés** (les erreurs du petit modèle contiennent celles du grand), ce qui est à peu près le
comportement réel d'un modèle distillé et donne à McNemar une direction bien définie.

Le remplacement temporaire du chemin des prédictions mérite un mot :

```python
original = report.PREDICTIONS_DIR
report.PREDICTIONS_DIR = predictions          # on pointe vers le dossier temporaire
try:
    rows, summary = report.analyse(rows, "bert-base", seed=42)
finally:
    report.PREDICTIONS_DIR = original         # on remet, quoi qu'il arrive
```

C'est un *monkey-patch* : on modifie une variable de module pour la durée d'un test. Le
`try/finally` garantit la restauration même si `analyse` lève une exception. (Ici la
restauration est sans danger : personne d'autre n'a besoin de la valeur temporaire. Comparer avec
l'histoire de `load_hf_evaluate`, où restaurer était précisément le bug.)

`test_analysis` vérifie aussi `dominance_lines` sur deux lignes fabriquées à la main — un
dominateur à p = 0,40 doit donner « latence seulement », un à p = 0,01 « les deux axes » — et que
`config.smoke_dirs()` ne pointe que sous `results/smoke/`.

---

### 3.9 `check_env.py` — le contrôle technique

Trois choses : lister les versions installées, décrire le matériel, conseiller un dimensionnement.

```python
def _version(package):
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None
```

`importlib.metadata.version` lit la version déclarée par le gestionnaire de paquets, sans
importer le module — donc sans payer les secondes de chargement de `transformers`.

L'import de `common` est **tardif** et commenté comme tel : `common` a besoin de `torch`, donc il
ne faut l'importer qu'après avoir constaté que `torch` est là. Sinon le diagnostic planterait au
lieu de dire ce qui manque — un outil de diagnostic doit survivre à ce qu'il diagnostique.

---

## 4. Les formules, rassemblées

| Quantité | Formule | Où |
|---|---|---|
| Rétention d'accuracy | $\text{acc}_m / \text{acc}_{\text{base}}$ | `report.analyse` |
| Écart en points | $100\,(\text{acc}_m - \text{acc}_{\text{base}})$ | `report.analyse` |
| Accélération | $\ell_{\text{base}} / \ell_m$ | `report.analyse` |
| Réduction de latence | $100\,(1 - \ell_m/\ell_{\text{base}}) = 100\,(1 - 1/\text{accél.})$ | `report.analyse` |
| Taux de change | $\dfrac{100\max(0,\ \text{acc}_{\text{base}}-\text{acc}_m)}{\text{accél.}-1}$ | `report.analyse` |
| Intervalle de Wilson | $\dfrac{\hat p + \frac{z^2}{2n}}{1+\frac{z^2}{n}} \pm \dfrac{z}{1+\frac{z^2}{n}}\sqrt{\dfrac{\hat p(1-\hat p)}{n} + \dfrac{z^2}{4n^2}}$ | `common.wilson_interval` |
| McNemar exact | $p = 2\sum_{i=0}^{\min(b,c)}\binom{b+c}{i}2^{-(b+c)}$ | `common.mcnemar_exact` |
| Bootstrap apparié | percentiles 2,5 et 97,5 de $\{\overline{a_{I_k}} - \overline{b_{I_k}}\}_{k=1}^{10^4}$, $I_k$ tirés avec remise | `common.paired_bootstrap_delta` |
| Dominance de Pareto | $j \succ i \iff \text{acc}_j \ge \text{acc}_i \wedge \ell_j \le \ell_i \wedge (\text{acc}_j > \text{acc}_i \vee \ell_j < \ell_i)$ | `common.pareto_front` |

Deux ordres de grandeur à garder en tête :

- **Résolution de l'évaluation.** Avec $n$ exemples, l'écart type d'une accuracy seule vaut
  environ $\sqrt{\hat p(1-\hat p)/n}$, soit $\approx 0{,}67$ point à $n = 2000$ et
  $\hat p = 0{,}90$. Mais pour **comparer** deux modèles sur les mêmes exemples, ce qui compte est
  la part $d$ d'exemples sur lesquels ils sont en désaccord (les discordants de McNemar) : l'écart
  type de la différence vaut à peu près $\sqrt{d/n}$, et un écart devient détectable vers
  $2\sqrt{d/n}$. Dans les résultats committés, $d$ vaut de 2,7 à 5,5 % (de 53 à 110 discordants
  sur 2000 contre BERT-base), d'où $2\sqrt{d/n}$ entre 0,7 et 1 point — disons $\approx 1$ point. Un écart inférieur à $\approx 1$ point n'est donc pas
  résoluble à cette taille. (Attention au raccourci $1/\sqrt n$ : à $n = 2000$ il donne 2,2
  points, ce qui ne correspond à rien ici.) C'est ce que dit la colonne McNemar, et c'est pourquoi
  le rapport insiste sur « pas de preuve d'écart » plutôt que « pas d'écart ».
- **Variance de fine-tuning.** Deux entraînements du même modèle à seeds différentes diffèrent de
  quelques dixièmes de point — le même ordre que les petits écarts du tableau. D'où
  `--seeds 42 43 44` et la section « Seed-to-seed variation ».

---

## 5. Pour t'approprier le code : cinq exercices

Du plus simple au plus instructif. Chacun se vérifie avec `python selftest.py`.

**1. Ajouter un modèle** (5 min, aucune autre modification)

Le registre contient déjà dix-sept modèles, donc commence par vérifier que l'extensibilité est
réelle : lance le sweep qui contient le contre-exemple le plus instructif (les résultats iront
sous `results/smoke/`, sans toucher aux vrais).

```bash
uv run python run_benchmark.py --sweep mechanisms --smoke
```

ALBERT-base y figure. Il partage ses poids **entre les couches**, donc il ne compte que ~12 M
paramètres — mais il garde 12 couches de largeur 768, donc il effectue *exactement le même calcul*
que BERT-base. Il doit donc ressortir **petit et pas plus rapide**, et sortir de la frontière de
Pareto. C'est la démonstration que « moins de paramètres » et « plus rapide » sont deux propriétés
distinctes : le partage de poids économise la mémoire, pas les multiplications de matrices.

Puis ajoute-en un toi-même. Une entrée dans `MODELS` suffit, par exemple :

```python
"electra-base": ModelSpec(
    slug="electra-base", checkpoint="google/electra-base-discriminator",
    label="ELECTRA-base", lr=3e-5, family="ELECTRA",
    note="12 couches, hidden 768. Le pendant taille normale d'ELECTRA-small.",
),
```

Rien d'autre à toucher : tableaux, statistiques, figures et couleurs le prennent en compte. C'est
le test qu'un découpage est bon.

**2. Changer la baseline** (1 min)

`python report.py --baseline distilbert`. Toutes les colonnes relatives se recalculent. Avant de
lancer, prédis ce que tu vas voir : dans les résultats committés, DistilBERT (90,20 %) est
légèrement **meilleur** que BERT-base (89,95 %). La rétention de BERT-base passe donc **sous**
100 % (≈ 99,7 %), son accélération sous 1× — et son taux de change est vide, puisque le
dénominateur est négatif. Vérifie aussi que McNemar donne le même p qu'avant (le test est
symétrique), et regarde quel modèle le verdict recommande désormais. C'est un bon test de
compréhension de `analyse`.

(Attention : le rapport par défaut est réécrit. Relance `python report.py` ensuite pour revenir à
la baseline BERT-base.)

**3. Ajouter une métrique** (20 min, touche deux fichiers)

Ajouter le coefficient kappa de Cohen. Le chemin complet :

- `common.compute_metrics` : calculer et renvoyer `"cohen_kappa"`
- `report.CSV_COLUMNS` : ajouter la colonne
- `report.detail_table` : ajouter l'en-tête et la cellule

Cet exercice te fait traverser tout le pipeline et montre où sont les points de couplage.

**4. Mesurer l'effet de la longueur de séquence** (30 min de calcul)

```bash
python run_benchmark.py --max-length 64  --smoke
python run_benchmark.py --max-length 256 --smoke
```

Le second run ne réutilise pas les checkpoints du premier : `max_length` fait partie de
l'empreinte, donc `train.py` affiche `[retrain] …: configuration changed — max_length: 64 -> 256`
et réentraîne. (L'ancienne version les aurait réutilisés sans rien dire.)

L'attention est quadratique en longueur, le feed-forward linéaire. Le classement en vitesse peut
donc **changer** avec `--max-length` : MiniLM est étroit (donc son feed-forward est bon marché)
mais profond avec 12 couches d'attention. À grande longueur, l'attention domine et son avantage
se réduit. C'est une vraie expérience, pas une simple relecture de code.

**5. Casser un test exprès** (5 min, très instructif)

Dans `common.mcnemar_exact`, remplacer `2.0 * tail` par `tail` (test unilatéral au lieu de
bilatéral). Lancer `python selftest.py` : le check « McNemar exact matches the hand-computed
binomial tail » doit échouer, et le code de retour doit être 1. Ça te montre que le filet de
sécurité attrape réellement quelque chose — un test qui ne peut pas échouer ne teste rien.

---

## 6. Où regarder quand quelque chose ne va pas

| Symptôme | Piste |
|---|---|
| `No results for dataset 'ag_news' seed 42` | `evaluate.py` n'a pas tourné, ou pas sur cette seed |
| `no checkpoint for X … Run train.py first` | `train.py` n'a pas tourné pour ce modèle |
| Métriques via `sklearn` alors que `evaluate` est installé | voir `common.load_hf_evaluate` ; l'échec est imprimé avec son type d'exception |
| `[compat] TrainingArguments: ignoring unsupported [...]` | normal — la version de `transformers` n'a pas ces drapeaux |
| `[retrain] X: configuration changed — …` | normal : le checkpoint existant a été entraîné avec d'autres réglages (voir §3.3) ; la raison est listée |
| `report.py` ne trouve pas les résultats d'un run `--smoke` | ils sont sous `results/smoke/` : `python report.py --smoke` |
| Latences très variables entre relances | `--num-threads` non fixé, ou machine chargée ; regarde la colonne « p50 range over rounds » et augmente `--latency-rounds` |
| `machine NOT settled after 45s` | quelque chose d'autre tourne, ou la machine est thermiquement instable — les comparaisons de temps ne sont pas fiables |
| Avertissement « newly initialized » au chargement | normal : la tête de classification est neuve, c'est ce qu'on entraîne |
| Les figures ne changent pas après édition | `report.py` réécrit `plots/` ; vérifie que tu n'as pas passé `--no-plots` |

Et le réflexe général : `results/environment.json` enregistre le matériel, les versions, le
nombre de threads et — écrit par `evaluate.py` — l'état de la mise en régime (`settle_converged`,
`settled_gflops`) et le nombre de tours du dernier run. C'est le premier fichier à ouvrir quand deux séries de mesures
ne se ressemblent pas.
