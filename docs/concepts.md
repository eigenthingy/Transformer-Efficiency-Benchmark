# Comprendre le projet : les concepts

Ce document explique le **domaine** : ce qu'est un encodeur transformer, ce que fait la
tokenization, ce que veut dire fine-tuner, et ce que le benchmark compare exactement.

Le compagnon [`guide-lecture.md`](guide-lecture.md) explique le **code**. Celui-ci n'en suppose
aucune connaissance.

---

## 1. La tâche

Le benchmark tourne sur **AG News** : des dépêches de presse à ranger dans quatre catégories.

| Texte d'entrée | Étiquette attendue |
|---|---|
| *"Oil prices soar to all-time record, posing new menace to US economy"* | Business |
| *"Dream Team sputters to Olympic bronze"* | Sports |
| *"Researchers find new way to fold proteins"* | Sci/Tech |

C'est de la **classification de texte** : une entrée, une sortie parmi quatre. Volontairement
simple, parce que le sujet du projet n'est pas la tâche — c'est le **coût** des modèles qui la
résolvent.

Un point d'honnêteté que le README répète : cette tâche est facile. Un modèle à sac de mots
atteint déjà ~90 %. La marge entre un gros et un petit transformer y est donc **comprimée**. Sur
de l'inférence multi-phrases ou du question-réponse, les écarts seraient plus larges. C'est une
limite du protocole, pas un résultat.

---

## 2. De texte à nombres : la tokenization

Un réseau de neurones ne traite que des nombres. Il faut donc transformer `"Oil prices soar"` en
une suite d'entiers. C'est le rôle du **tokenizer**, et chaque modèle a le sien, fixé à son
pré-entraînement — on ne peut pas les échanger.

### 2.1 Le vocabulaire et les sous-mots

Le tokenizer possède un **vocabulaire** : une liste figée de morceaux de texte, typiquement 30 000
à 128 000 entrées. Chaque morceau a un indice.

Pourquoi des *morceaux* et pas des mots entiers ? Parce qu'une liste de mots entiers serait à la
fois gigantesque et toujours incomplète — le premier nom propre inconnu la casserait. Les
tokenizers découpent donc en **sous-mots** :

```
"tokenization"   →  ["token", "##ization"]
"Oil prices soar" → ["oil", "prices", "soar"]
"Ouagadougou"    →  ["ou", "##aga", "##dou", "##go", "##u"]
```

Le `##` signifie « collé au morceau précédent ». Avantage décisif : **tout** texte est
représentable, même un mot jamais vu, parce qu'on peut toujours retomber sur des morceaux plus
courts, et à la limite sur des caractères.

Trois algorithmes de découpage cohabitent dans ce registre, et c'est pour ça que les tokenizers ne
sont pas interchangeables :

| Algorithme | Modèles du registre |
|---|---|
| **WordPiece** | BERT, DistilBERT, MiniLM, ELECTRA, BERT-mini, BERT-tiny |
| **BPE** (Byte-Pair Encoding) | RoBERTa, DistilRoBERTa, ModernBERT, Ettin |
| **SentencePiece / Unigram** | DeBERTa-v3, ALBERT |

Conséquence observable dans la sortie du run : sur les **mêmes** articles (le taux est estimé sur
les 512 premiers exemples d'évaluation), MiniLM annonce `truncated=0.4%` et Ettin-32M
`truncated=0.6%`, soit 2 et 3 articles sur 512. Le second découpe un peu plus finement, donc
produit un peu plus de tokens, donc dépasse la limite un peu plus souvent.

### 2.2 Les tokens spéciaux

Au découpage s'ajoutent des marqueurs :

- **`[CLS]`** en tête. Ce n'est pas du texte : c'est un emplacement dont le vecteur final servira
  de résumé de la phrase entière. C'est lui qu'on branche sur le classifieur.
- **`[SEP]`** en fin de segment.
- **`[PAD]`** pour remplir jusqu'à la longueur voulue.

### 2.3 Ce que le tokenizer renvoie

```python
tokenizer("Oil prices soar", truncation=True, padding="max_length", max_length=128)
```

```
input_ids      = [101, 3514, 7597, 25555, 102, 0, 0, 0, ... ]   ← 128 entiers
attention_mask = [  1,    1,    1,     1,   1, 0, 0, 0, ... ]   ← 128 zéros/uns
```

- **`input_ids`** : les indices dans le vocabulaire.
- **`attention_mask`** : 1 pour un vrai token, 0 pour du remplissage. Sans lui, le modèle
  « regarderait » le padding comme du contenu.
- **`token_type_ids`** : présent chez BERT (qui distingue deux segments), absent chez DistilBERT.
  Le code gère les deux cas sans condition explicite, parce que chaque modèle tokenise avec son
  propre tokenizer.

### 2.4 Troncature et padding, et pourquoi ce choix

Deux réglages, deux décisions méthodologiques.

**`truncation=True, max_length=128`** — au-delà de 128 tokens, on coupe. C'est le « budget de
tokens ». Sur AG News il suffit presque toujours (d'où les 0,4 %). Sur IMDB, dont les critiques
sont longues, il faudrait 256 et la troncature serait massive — le rapport l'indiquerait, et la
colonne accuracy devrait alors se lire « accuracy à ce budget », pas « accuracy plafond ».

**`padding="max_length"`** — on remplit *tout* à 128, même une dépêche de 20 tokens. C'est
délibéré et contre-intuitif, parce que c'est du calcul gaspillé. La raison : tous les modèles
traitent alors des tenseurs de forme **identique**. La colonne latence compare donc des
architectures et non des hasards de géométrie de batch.

Le coût de ce choix est réel et il est écrit dans le README : il désavantage ModernBERT et Ettin,
dont l'architecture sait précisément *éviter* de calculer sur du padding. Le protocole les mesure
donc dans leur configuration la moins favorable.

### 2.5 Pourquoi la taille du vocabulaire compte pour le coût

Les indices doivent devenir des vecteurs. C'est la **table d'embeddings** : une matrice de
`vocabulaire × largeur`, où l'on lit une ligne par token.

| Modèle | Vocabulaire | Largeur | Table d'embeddings |
|---|---:|---:|---:|
| BERT-base | 30 522 | 768 | 23,4 M paramètres |
| DeBERTa-v3-xsmall | 128 100 | 384 | **49,2 M paramètres** |

DeBERTa-v3-xsmall a un encodeur d'environ 21 M paramètres — mais sa table d'embeddings en fait
49 M, soit **~70 % du modèle**. Or lire une ligne dans une table n'est pas un calcul : c'est un
accès mémoire, quasi gratuit en temps.

D'où la mesure séparée dans tout le projet : `params_embedding` et `params_non_embedding`. C'est
le seul moyen d'éviter la conclusion fausse « DeBERTa-v3-xsmall est gros donc lent ». Il est gros
en mémoire et léger en calcul.

Le même piège en miroir : BERT-base et DistilBERT portent **la même** table d'embeddings (~23,8 M
en comptant les embeddings de position). Donc « DistilBERT a 40 % de paramètres en moins » sous-
estime la compression réelle : c'est **49 %** de l'encodeur, et c'est l'encodeur qui coûte du
temps.

---

## 3. Ce qu'est un encodeur transformer

### 3.1 La structure

```
input_ids (128 entiers)
      ↓  table d'embeddings  →  128 vecteurs de dimension d
      ↓
  ┌───────────────────────┐
  │  couche 1             │
  │   · self-attention    │   ← les tokens se regardent entre eux
  │   · feed-forward      │   ← chaque token est transformé isolément
  └───────────────────────┘
      ↓
     ...  (L couches identiques en structure, poids distincts)
      ↓
  128 vecteurs de dimension d
      ↓  on prend celui de [CLS]
      ↓  couche linéaire  d → 4
   4 scores (logits) → la classe prédite est l'indice du plus grand
```

Deux nombres décrivent l'essentiel :

- **`L`** = le nombre de couches, la **profondeur**.
- **`d`** = `hidden_size`, la dimension des vecteurs, la **largeur**.

BERT-base est `L=12, d=768`. MiniLM-L12 est `L=12, d=384` : même profondeur, moitié de largeur.
DistilBERT est `L=6, d=768` : même largeur, moitié de profondeur. **Ces deux-là compriment le
même modèle dans deux directions orthogonales** — c'est la comparaison la plus instructive du
registre.

### 3.2 La self-attention, sans les mathématiques

Le problème que l'attention résout : le sens d'un mot dépend des autres. Dans *« Oil prices
soar »*, `soar` doit informer la représentation de `prices`.

Le mécanisme : chaque token produit trois vecteurs — une **requête** (ce que je cherche), une
**clé** (ce que je propose), une **valeur** (ce que je transmets). Pour chaque paire de tokens on
calcule la compatibilité requête-clé, on normalise ces scores en poids sommant à 1, et chaque
token reçoit la moyenne pondérée des valeurs des autres. Un token « regarde » donc toute la
phrase, plus ou moins fort selon la pertinence.

Les **têtes** (`num_attention_heads`) font ce calcul plusieurs fois en parallèle sur des
sous-espaces différents, ce qui permet à une tête de suivre la syntaxe et à une autre la
coréférence.

Ce qu'il faut en retenir pour le benchmark : le calcul se fait sur **toutes les paires** de
tokens. Son coût croît donc en `T²` avec la longueur `T`, alors que le reste croît linéairement.
C'est pourquoi le classement en vitesse peut changer avec `--max-length`.

### 3.3 Le feed-forward

Après l'attention, chaque token passe indépendamment dans un petit réseau : `d → intermediate →
d`, avec `intermediate` classiquement `4×d`. C'est là que se trouve **la majorité des paramètres
et du calcul**.

Un détail qui a corrigé une de mes estimations pendant ce projet : ModernBERT et Ettin utilisent
un feed-forward **GeGLU**, dont l'`intermediate_size` est bien plus petit — 1152 au lieu de 3072
pour la même largeur 768. ModernBERT-base a 22 couches contre 12 à BERT-base, mais son
feed-forward étroit compense : il coûte ~1,3× BERT-base par token, pas 1,8× comme sa profondeur le
suggérait. **La profondeur seule ne prédit pas le coût.**

### 3.4 Le coût, en une formule

À longueur de séquence fixe, le coût d'une passe avant est dominé par les multiplications de
matrices, et vaut approximativement

$$\text{coût} \;\propto\; L \cdot d^2$$

C'est le « proxy FLOP » que le notebook compare aux latences mesurées. La colonne
`measured / predicted` de cette analyse dit où le modèle est **plus lent que son arithmétique** :
l'écart, c'est le surcoût qui ne rétrécit pas avec la largeur — dispatch Python, lancements de
noyaux, trafic mémoire.

C'est une prédiction déjà vérifiée par les mesures de ce projet : diviser la **profondeur** par
deux (DistilBERT) rend presque 2× ; diviser la **largeur** par deux (MiniLM) ne rend que ~3,1×
alors que l'arithmétique promettait 4×. Des matmuls étroites exploitent moins bien le processeur.

### 3.5 Ce que « pré-entraîné » veut dire

Un encodeur brut est inutile. Ce qui le rend utile, c'est un **pré-entraînement** sur d'énormes
quantités de texte non étiqueté, via une tâche qui ne demande aucune annotation humaine :

- **Masked Language Modelling** (BERT, RoBERTa, DistilBERT, ModernBERT, Ettin) : on masque ~15 %
  des tokens et le modèle doit les retrouver. Pour deviner *« Oil ___ soar »*, il faut avoir appris
  de la syntaxe, de la sémantique et des faits.
- **Replaced Token Detection** (ELECTRA) : un petit générateur remplace certains tokens par des
  candidats plausibles, et le modèle doit dire *pour chaque token* s'il est original ou substitué.
  Le signal porte sur 100 % des positions au lieu de 15 %, ce qui rend l'entraînement bien plus
  efficace à taille donnée. C'est pour ça qu'ELECTRA-small est dans le registre : il représente
  « petit grâce à un meilleur objectif », pas « petit par compression ».

---

## 4. Le fine-tuning : ce que `train.py` fabrique

### 4.1 Le principe

On part de l'encodeur pré-entraîné, on lui greffe une **tête de classification** — une simple
couche linéaire `d → 4` — et on entraîne le tout sur les 4000 exemples étiquetés.

La tête est initialisée **au hasard**, puisqu'elle n'existe pas dans le checkpoint. D'où
l'avertissement qui apparaît dans la sortie :

```
Some weights of BertForSequenceClassification were not initialized ...
and are newly initialized: ['classifier.bias', 'classifier.weight']
```

Ce n'est pas une erreur, c'est la description exacte de ce qu'on fait.

### 4.2 La boucle d'entraînement

```
4000 exemples ÷ batch de 16 = 250 pas    ← le « 0/250 » de la barre de progression
```

Chaque pas :

1. **Passe avant** — 16 exemples traversent le modèle, on obtient 16 × 4 logits.
2. **Perte** — l'entropie croisée mesure l'écart aux étiquettes réelles. Un seul nombre.
3. **Passe arrière** — la rétropropagation calcule le gradient de la perte par rapport à chacun
   des 109 millions de paramètres.
4. **Mise à jour** — l'optimiseur AdamW déplace chaque paramètre un peu à l'opposé de son
   gradient.

La passe arrière coûte environ deux fois la passe avant, d'où le facteur ~3 entre inférence et
entraînement.

### 4.3 Les hyper-paramètres, et pourquoi ces valeurs

**`learning_rate = 3e-5`** — très petit. En entraînement depuis zéro on utiliserait `1e-3`. Ici on
ne veut pas apprendre : on veut **déplacer légèrement** des poids qui savent déjà beaucoup. Un LR
trop grand détruit la connaissance pré-entraînée (« oubli catastrophique »).

Les modèles plus petits tolèrent — et réclament — un LR plus grand : `5e-5` pour DistilBERT et
MiniLM, `1e-4` pour ELECTRA-small, `3e-4` pour BERT-tiny. C'est pour ça que `config.py` porte un
LR par modèle, et c'est une **décision méthodologique déclarée** : elle mesure « le meilleur
atteignable par architecture ». `--uniform-lr` mesure l'autre chose — « le comportement à réglage
constant ». Deux questions différentes, deux réponses possibles ; le rapport dit laquelle a tourné.

**`epochs = 1`** — une seule passe sur les données. Suffisant pour cette tâche, et c'est ce qui
rend le benchmark faisable en CPU.

**`warmup_ratio = 0.1`** — le LR monte de 0 à sa valeur sur les 10 premiers pourcents, puis
décroît. Les premiers pas sont bruyants, la tête étant aléatoire ; démarrer doucement évite
d'abîmer l'encodeur.

**`weight_decay = 0.01`** — pénalise les poids de grande amplitude, une régularisation.

---

## 5. Ce que le benchmark compare, et pourquoi chaque mesure

### 5.1 La qualité

| Mesure | Ce qu'elle dit |
|---|---|
| **Accuracy** | proportion de bonnes réponses |
| **F1 macro** | moyenne non pondérée des F1 par classe — attrape un modèle qui gagne en accuracy en négligeant une classe rare |
| **Intervalle de Wilson à 95 %** | l'incertitude due au fait que 2000 exemples sont un **échantillon** |

Ce dernier est le point important. `accuracy=0.8865` n'est pas une propriété du modèle, c'est une
**estimation** obtenue sur 2000 exemples tirés au hasard. Avec 2000 autres exemples, on obtiendrait
un nombre légèrement différent. L'intervalle `[0.8719, 0.8997]` quantifie ce flou. Un classement
sans intervalle n'est pas une preuve.

### 5.2 Le coût

| Mesure | La question qu'elle répond |
|---|---|
| **Temps d'entraînement** | à quelle vitesse puis-je itérer sur mon problème ? |
| **Latence à batch = 1**, médiane et p95 | combien attend un utilisateur pour **une** requête ? |
| **Débit à batch = 32** | combien coûte le scoring d'un corpus entier ? |
| **Paramètres** (total / hors embeddings) | mémoire vs calcul — voir §2.5 |
| **Taille sur disque** | déploiement : image conteneur, démarrage à froid, embarqué |

**Latence et débit ne se déduisent pas l'un de l'autre.** Dans le run de ce projet, MiniLM-L12 fait
30 ms à batch 1 (donc ~33 ex/s) et 52,7 ex/s à batch 32 : grouper les exemples rapporte ici ~1,6×,
pas 32×. Sur CPU, le gain du batching est modeste et varie d'un modèle à l'autre (BERT-base passe
de ~11 à 16 ex/s, Ettin-32M de ~48 à 76) ; sur GPU il est massif. Selon la machine, il peut même
devenir nul ou négatif — bande passante mémoire saturée, baisse de fréquence sous charge AVX
soutenue. C'est exactement pourquoi les deux sont mesurés séparément au lieu d'en calculer un
depuis l'autre.

**Médiane plutôt que moyenne**, et p95 en plus : une pause de l'OS ou du ramasse-miettes déplace
une moyenne, pas une médiane. Et c'est le p95 qui décide si un modèle tient dans un budget de
requête — un service ne casse pas sur son cas moyen. Dans le run publié, la queue est courte
(MiniLM-L12 : p50 = 30,0 ms, p95 = 34,6 ms) parce que la machine était calme ; sur une machine
chargée, le p95 peut s'éloigner bien davantage du p50.

### 5.3 Les quantités relatives, et pourquoi elles seules voyagent

Les temps absolus dépendent de la machine. Ce qui se transporte **le mieux**, ce sont les
rapports — mais pas parfaitement : un petit modèle passe proportionnellement plus de temps en
surcoûts fixes (dispatch, lancements de noyaux), et un matériel différent ne valorise pas de la
même façon la profondeur et la largeur. Une accélération mesurée à 4,5× ici peut valoir 3×
ailleurs, et deux modèles de coût voisin mais de forme différente peuvent échanger leur place. Le
README détaille pourquoi (« Why these results depend on the machine »).

| Quantité | Formule |
|---|---|
| Rétention d'accuracy | acc(modèle) / acc(baseline) |
| Accélération | latence(baseline) / latence(modèle) |
| Réduction de latence | 1 − 1/accélération |
| **Taux de change** | points d'accuracy perdus ÷ (accélération − 1) |

Le dernier est la contribution propre du projet. Il répond à la question qu'une décision
d'ingénierie pose réellement : **combien de points d'accuracy coûte chaque × de vitesse
supplémentaire ?** Le `−1` au dénominateur est essentiel : à `1×` on n'a rien gagné, le
dénominateur doit s'annuler.

### 5.4 La frontière de Pareto

Avec deux critères — accuracy à maximiser, latence à minimiser — il n'y a pas de « meilleur »
unique. Un modèle est **dominé** si un autre est au moins aussi bon sur les deux et strictement
meilleur sur au moins un. Les non dominés forment la **frontière**.

Sortir de la frontière est un verdict sec : il existe un autre modèle plus précis **et** plus
rapide, donc plus aucun argument. ALBERT-base est dans le registre exprès pour être ce cas :
~12 M paramètres mais le calcul complet d'un BERT-base, parce que partager les poids entre couches
économise la mémoire et pas les multiplications de matrices. Toute analyse qui confond
« paramètres » et « coût » est réfutée par cette seule ligne.

---

## 6. Les mécanismes : huit façons d'être moins cher

C'est l'axe qui structure le registre. Empiler des modèles n'apprend rien ; comparer des
**mécanismes** apprend quelque chose.

**Distillation de connaissance** — *DistilBERT, DistilRoBERTa.* On entraîne un petit modèle
(l'élève) à reproduire non pas les étiquettes, mais la **distribution de probabilités** du gros
(le professeur). Signal beaucoup plus riche : « 70 % Business, 25 % Sci/Tech » transmet une
information sur la proximité des classes qu'une étiquette dure n'a pas. DistilBERT retire une
couche sur deux de BERT-base et rattrape par ce signal.

**Distillation d'attention** — *MiniLM.* L'élève imite les **matrices d'attention** et les
relations entre valeurs du professeur, et non ses sorties. Comme ces objets ne dépendent pas de la
largeur, l'élève peut être plus **étroit** que le professeur, pas seulement moins profond. D'où le
« deep and thin » : profondeur conservée, largeur divisée.

**Meilleur objectif de pré-entraînement** — *ELECTRA-small.* Voir §3.5 : signal sur 100 % des
tokens au lieu de 15 %.

**Partage de poids** — *ALBERT-base.* Les 12 couches partagent **un seul** jeu de poids. Effondre
le nombre de paramètres, ne change rien au calcul. Le contre-exemple pédagogique du registre.

**Pré-entraîné petit** — *BERT-mini, BERT-tiny.* Pas de compression : on pré-entraîne directement
une petite architecture. BERT-tiny (2 couches, largeur 128) est là pour montrer **où la courbe
d'accuracy casse** — seul moyen de savoir que les autres étaient sur sa partie plate.

**Architecture moderne** — *ModernBERT, Ettin.* Six ans d'améliorations cumulées : RoPE au lieu
d'embeddings de position appris, GeGLU, attention alternée locale/globale, suppression des biais,
et surtout beaucoup plus de données d'entraînement. Ettin est cette architecture publiée en échelle
complète (17 M à 400 M), ce qui permet des comparaisons **à taille appariée** avec les anciens
modèles distillés.

**Vocabulaire large et attention désenchevêtrée** — *DeBERTa-v3.* Vocabulaire de 128k et un
mécanisme d'attention qui traite contenu et position séparément. Excellent en accuracy par
paramètre d'encodeur, au prix d'une table d'embeddings énorme.

**Même forme, meilleur entraînement** — *RoBERTa-base.* Architecturalement identique à BERT-base,
entraîné plus longtemps sur plus de données. Isole la recette d'entraînement de l'architecture :
même calcul, même forme, accuracy différente.

---

## 7. Pourquoi les statistiques, concrètement

Reprends les deux nombres du run : MiniLM-L12 à **88,65 %**, Ettin-32M à **89,25 %**. Ettin
gagne-t-il ?

**On ne peut pas le dire depuis ces deux nombres.** L'intervalle de Wilson de MiniLM est
`[87,2 % ; 90,0 %]`, et 89,25 % est dedans. Trois outils permettent de trancher.

**L'intervalle de Wilson** donne l'incertitude d'**une** accuracy. Il est préféré à
l'approximation normale parce qu'il reste dans `[0,1]` et garde sa couverture quand l'accuracy est
proche de 1 — le régime où l'on travaille.

**Le test exact de McNemar** compare **deux** classifieurs sur le même jeu de test. Son idée
centrale : on ne compare pas deux nombres, on regarde les **paires**. Les exemples que les deux ont
bons, ou que les deux ont faux, n'apportent aucune information sur lequel est meilleur — on les
écarte. Reste les **discordants** : ceux qu'un seul a bons. Sous l'hypothèse « les deux sont
équivalents », chaque discordance est un tirage à pile ou face. Le test calcule la probabilité
d'observer un déséquilibre au moins aussi marqué par hasard.

**Le bootstrap apparié** donne un intervalle sur la **différence**. On ré-échantillonne 10 000
fois les exemples avec remise, en utilisant **les mêmes indices pour les deux modèles**. L'écart
qui en ressort ne mélange pas deux incertitudes indépendantes : il isole le désaccord entre les
deux modèles. C'est pourquoi cet intervalle est bien plus serré que la différence de deux
intervalles de Wilson — et pourquoi cette dernière serait une erreur.

**Ce qu'un résultat non significatif veut dire.** Le rapport écrit « pas de preuve de différence »,
jamais « pas de différence ». Pour une comparaison appariée, le plus petit écart détectable est de
l'ordre de `2·√(d/n)`, où `d` est la part d'exemples sur lesquels les deux modèles ne sont pas
d'accord. Ici `d` vaut 3 à 6 %, soit ~1 point à `n = 2000`. Un résultat non significatif est donc en partie une
information sur la **taille de ton jeu d'évaluation**, pas seulement sur les modèles.

**Et la variance des seeds.** Deux entraînements du même modèle à graines différentes diffèrent de
quelques dixièmes de point — le même ordre que les petits écarts du tableau. D'où `--seeds 42 43
44` et la section « Seed-to-seed variation ». Sans elle, « DistilBERT perd 0,4 point » est
indiscernable de « mes runs varient de 0,4 point ».

---

## 8. Lire la sortie du run

```
[retrain] bert-base: configuration changed — max_train_samples: 200 -> 4000
```
Un checkpoint existe (du smoke test) mais il a été entraîné sur 200 exemples. Le garde-fou
d'empreinte refuse de le réutiliser. Sans ce contrôle, le run réel évaluerait silencieusement des
modèles entraînés sur un vingtième des données.

```
layers=12 hidden=768 heads=12 | params=109.5M (embeddings 23.8M)
train examples=4000 max_length=128 truncated=0.4% lr=3e-05 epochs=1.0 bs=16 device=cpu threads=12
```
L'inventaire complet, écrit avant la mesure pour qu'un lecteur sache exactement ce qui a été
mesuré.

```
0%|          | 0/250 [00:00<?, ?it/s]
```
Zéro pas sur 250, soit `4000 ÷ 16`. À ~4,5 s le pas pour BERT-base sur la machine du run publié,
la barre semble figée un moment avant d'avancer.

```
-> train time 1122.7s | 3.6 samples/s | loss 0.4875 | 418 MB on disk
```
Le chronomètre n'entoure que `trainer.train()` — pas le chargement, pas la tokenization, pas
l'écriture disque.

```
accuracy=0.8865 f1_macro=0.8874 (evaluate)  95% CI [0.8719, 0.8997]
```
La précision de MiniLM-L12 et son intervalle de Wilson (§5.1), calculés une fois, avant toute
mesure de temps : l'accuracy est déterministe, l'ordre n'y change rien.

```
MiniLM-L12         acc=88.65%  p50=  30.0ms  (range 29.4-31.2, spread 1.06x)
```
La latence retenue est la **médiane sur les rounds entrelacés** ; la plage dit de combien elle a
bougé d'un round à l'autre. 6 % ici : la machine était stable. Au-delà de 25 %, le rapport
l'annonce en tête, avant les tableaux.

---

## 9. Ce que ce projet ne mesure pas

À savoir avant de généraliser un résultat :

- **Une seule tâche.** La rétention sur de la classification de dépêches ne se transporte pas à de
  l'inférence logique ou du question-réponse. La marge y est comprimée (§1).
- **Une seule longueur de séquence.** L'attention est quadratique en longueur, le feed-forward
  linéaire : le classement en vitesse peut bouger avec `--max-length`.
- **Pas de recherche d'hyper-paramètres.** Une époque au LR recommandé, pas un optimum par modèle.
  Une recherche favoriserait probablement les petits modèles, moins chers à régler.
- **fp32, un seul appareil.** Quantisation, ONNX Runtime ou un GPU déplaceraient la colonne latence
  et pourraient en changer l'ordre. `results/environment.json` enregistre ce qui a réellement
  tourné.
- **La distillation n'est pas gratuite.** DistilBERT et MiniLM ont coûté un pré-entraînement
  chacun. Ce benchmark chiffre le fait d'**utiliser** un petit modèle, pas de le **créer**.
- **Le padding fixe désavantage les architectures modernes** (§2.4). Leur ligne se lit comme un
  plancher.
