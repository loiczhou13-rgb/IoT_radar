# Plan — robustesse aux mouvements parasites (chaîne SISO)

> **Version 1**, 2026-10-08. Branche `siso-detection`, créée depuis `dev` = `5a6edd0`.
> Étape 1 : conception, aucune modification de code.
> Les chiffres de l'audit (§1) et de la comparaison des estimateurs de centre (§3.3) ont été calculés avec des scripts temporaires, hors du dépôt, sur le code de `dev`.

## Sommaire

0. [Décisions](#0-décisions)
1. [Audit de l'anti-repliement](#1-audit-de-lanti-repliement)
2. [Diagnostic : pourquoi le radar se met en pause](#2-diagnostic--pourquoi-le-radar-se-met-en-pause)
3. [Conception de M1 à M4](#3-conception-de-m1-à-m4)
4. [Paramètres par défaut](#4-paramètres-par-défaut)
5. [Simulation des perturbateurs](#5-simulation-des-perturbateurs)
6. [Protocole d'évaluation](#6-protocole-dévaluation)
7. [Hors périmètre : annulation spatiale et mouvement de la cible](#7-hors-périmètre--annulation-spatiale-et-mouvement-de-la-cible)
8. [Risques](#8-risques)
9. [Étapes et commits](#9-étapes-et-commits)

---

## 0. Décisions

**Validées** (réponses du 2026-10-08) :

| # | Décision |
|---|---|
| Q1 | La chaîne SISO passe à **2,45 GHz** (bande ISM, f_D ≈ 16·v), dans un commit séparé qui met aussi à jour les chiffres des README. |
| Q2 | Indice de contamination **normalisé par les largeurs de bande** (rapport de densités) : un bruit blanc vaut 0 dB. |
| Q3 | `feature/mimo-localization` est ignorée pour le moment. Le test qui fige la chaîne à une voie est écrit sur cette branche. |
| Q4 | Évaluation sur la simulation et les 69 anciennes sessions ; aucun enregistrement du problème constaté n'est disponible. |

**Prises par défaut, à contester si besoin :**

| # | Décision |
|---|---|
| D1 | M1 et M3 travaillent à une **fréquence Doppler intermédiaire de 100 Hz**. La cascade actuelle y passe déjà juste avant son dernier étage ×5 (§1) : la couper en deux ne change rien au calcul. |
| D2 | M1 à M3 sont **activés par défaut**. Les désactiver tous redonne la chaîne actuelle au bit près (test de référence). **Révisé après l'évaluation (réponse du 2026-10-09) : M2 est désactivé par défaut**, car M1+M3 fait au moins aussi bien que M1+M2+M3 dans toutes les scènes évaluées. |
| D3 | Les intervalles de perturbation simulés sont stockés dans **`/ground_truth/disturbances`** (début, longueur, type). Le vocabulaire des labels de `/annotations` ne change pas. |
| D4 | `CLAUDE.md` n'existe toujours pas. J'applique les critères de lisibilité de la première demande. |

---

## 1. Audit de l'anti-repliement

**Méthode** (Crochiere & Rabiner, 1981).
- Pour chaque étage de décimation (débit d'entrée F, facteur q, débit de sortie F/q), on calcule les bandes qui se replient dans la bande respiratoire [−f_b, +f_b] en sortie de l'étage : m·F/q ± f_b, pour m = 1 … ⌊(F/2 + f_b)/(F/q)⌋.
- On relève l'atténuation minimale du filtre de l'étage dans ces bandes. Les filtres sont des Tchebychev I d'ordre 8, ondulation 0,05 dB, Wn = 0,8/q.
- Un **suivi composite** propage chaque fréquence d'entrée à travers toute la cascade : gain cumulé et fréquence repliée finale. On en déduit le pire cas, quel que soit le chemin de repli.
- Les fréquences sont prises **après le mélangeur** : 0 Hz correspond à la porteuse de l'écho.

### 1.1 Chaîne actuelle : 1 MS/s → 20 Hz, facteur 50 000 = 2⁴·5⁵, f_b = 1 Hz

| Étage | Débits | Première bande repliée | Atténuation minimale dans les bandes repliées | Retard de groupe (DC) |
|---|---|---|---|---|
| ×2 (×4 étages) | 1 MS/s → 62,5 kS/s | F/2 ± 1 Hz | > 360 dB (zéro du filtre à Nyquist) | < 0,1 ms |
| ×5 | 62,5 kS/s → 12,5 kS/s | 12,5 kHz ± 1 Hz | 92,8 dB | 0,2 ms |
| ×5 | 12,5 kS/s → 2,5 kS/s | 2,5 kHz ± 1 Hz | 92,7 dB | 1,0 ms |
| ×5 | 2,5 kS/s → 500 Hz | 500 Hz ± 1 Hz | 92,6 dB | 4,8 ms |
| ×5 | 500 Hz → **100 Hz** | 100 Hz ± 1 Hz | 91,8 dB | 23,9 ms |
| ×5 | **100 Hz → 20 Hz** | **20 Hz ± 1 Hz** | **87,8 dB** (à 19 Hz) | 121,7 ms |

- **Pire cas composite :** 88,1 dB, pour une entrée à ±19 Hz repliée à ∓1 Hz.
- **Exigence de 60 dB : respectée partout**, avec 28 dB de marge. Un perturbateur à 19 Hz ressort à −88 dB : il n'apparaît pas comme une respiration.
- **Anciennes sessions converties** (2 kHz → 20 Hz, étages 2, 2, 5, 5) : mêmes conclusions, pire cas 87,9 dB à 19 Hz.

### 1.2 Ce qui passe sans repliement

Fréquence d'entrée → fréquence à 20 Hz → atténuation :

| Entrée | 3 Hz | 5 Hz | 8 Hz | 10 Hz | 12 Hz | 16 Hz | 19 Hz | 25 Hz | −503,5 Hz (DC du récepteur) |
|---|---|---|---|---|---|---|---|---|---|
| À 20 Hz | 3 | 5 | 8 | −10 | −8 | −4 | −1 | 5 | −3,5 |
| Atténuation | **0,4 dB** | **0,4 dB** | **0,4 dB** | 24,6 dB | 44,4 dB | 71,9 dB | 88,1 dB | 116 dB | 93,8 dB |

- La bande passante du dernier étage s'étend jusqu'à **8 Hz**. Les gestes (3 à 16 Hz) arrivent donc presque intacts dans le temps lent, alors qu'ils sont hors de la bande respiratoire.
- Ils ne se replient pas, mais ils traversent l'ajustement de cercle et l'arctangente, qui sont **non linéaires** (§2).

**Conclusion.** M1 n'a pas à corriger l'anti-repliement, qui est déjà conforme. Il doit retirer, **sur l'IQ et avant la démodulation**, tout ce qui est hors de la bande respiratoire mais sous la fréquence de Nyquist du temps lent. L'audit reste un test permanent : il vérifie que toute nouvelle configuration de décimation respecte 60 dB.

---

## 2. Diagnostic : pourquoi le radar se met en pause

Dans le temps lent :

```
x(t) = C(t) + a·e^{jφ(t)} + Σ_k b_k(t)·e^{jφ_k(t)}
```

1. **Les perturbateurs entrent dans le temps lent.** Les gestes (3 à 16 Hz) traversent la décimation (§1.2). Les balancements (0,8 Hz) et la marche (16 à 25 Hz à 2,45 GHz, 23 à 35 Hz à 3,5 GHz) y entrent aussi, la marche en partie.
2. **L'ajustement de cercle se trompe**, car les points ne sont plus sur l'arc de la cible.
3. **L'arctangente mélange tout**, car elle n'est pas linéaire :
   - arg(a·e^{jφ} + b·e^{jφ_k}) contient des produits φ ± φ_k ;
   - dès que |b| ≈ |a|, des sauts de cycle (2π) apparaissent, donc des marches dans la phase déroulée, dont le spectre est large et inclut la bande respiratoire.
4. **Le déplacement crête à crête dépasse 30 mm** → `motion = True` → état `MOTION` maintenu 3 s (`motion_hold_s`).

En milieu ouvert, où il y a toujours un peu d'activité, le radar est donc presque constamment en pause.

**Principe de la correction** (demande, §3). La somme est linéaire en IQ, l'arctangente ne l'est pas : on supprime d'abord, sur l'IQ, ce qui n'est pas respiratoire (M1), on suit le centre qui dérive (M2), on pondère au lieu de couper (M3), et on mesure pendant les silences (M4).

---

## 3. Conception de M1 à M4

### 3.1 Ordre des traitements

```
Block (ADC, 1 MS/s) ─► contrôle de saturation de l'ADC (|I| ou |Q| ≥ 2047)                       [pause]
  ─► Mixer (écho à 0 Hz)
  ─► Decimator A : 1 MS/s → 100 Hz (les 8 premiers étages actuels, inchangés)
        │
        ├─► M3a  indice de contamination par bloc de 1 s, AVANT M1 :
        │        densité hors bande [2 Hz ; 50 Hz] / densité dans la bande [0,1 ; 1 Hz]
        │        → poids w_k ∈ [0, 1], fonction lisse de l'indice (0 si bloc saturé)
        │        → activité hors bande (densité / son plancher) → « quelqu'un bouge »
        │
  ─► M1  passe-bas FIR complexe, phase linéaire, coupure 1,5 Hz, ≥ 60 dB dès 2 Hz, puis ×5 → 20 Hz
        (M1 désactivé : dernier étage Tchebychev ×5 actuel)
        poids retardés du retard du FIR, puis répétés pour chaque échantillon à 20 Hz
  ─► anneau du temps lent (fenêtre de 20 s, mise à jour toutes les 0,5 s), avec les poids
  ─► analyse de la fenêtre :
        dérive du LO → M2 centre glissant (Pratt sur des sous-fenêtres de 10 s, pas de 1 s,
        interpolation linéaire de C(t)) → arctangente → déplacement
        (M2 désactivé : un seul ajustement géométrique sur la fenêtre, comme aujourd'hui)
        → M3b anomalies dans la bande (crête à crête local sur 3 s > 30 mm) et excursions
          → poids 0 ; attribution à un passant si activité hors bande dans les 10 s avant
        → périodogramme de Lomb-Scargle pondéré (un décalage par segment) → fréquence, SNR,
          concentration → confiance × poids moyen
        → anomalie de la cible en cours → mouvement de la cible                     [pause]
  ─► BreathingDetector : pause seulement sur saturation ou anomalie dans la bande (M3 activé)
  ─► M4  mode mesure : fenêtre de 30 s lancée par l'opérateur → décision + résumé de qualité
  ─► PipelineOutput (+ poids, part contaminée, état de la mesure) ─► tableau de bord
```

- **Couper la cascade en deux** (Decimator A, puis M1 ou le dernier étage) fait les mêmes opérations dans le même ordre, à condition de garder la sortie intermédiaire en complex128. Avec M1 à M3 désactivés, la sortie est **identique au bit près** à l'actuelle ; le test de référence le vérifie (Q3).
- **Place dans la structure :**

| Élément | Module | Contenu |
|---|---|---|
| M1 | `iot_radar/dsp/doppler.py` (nouveau) | Conception du FIR passe-bas (Kaiser), filtre-décimateur à état (`DopplerLowpass`), audit d'anti-repliement (`alias_audit`) d'une cascade |
| | `iot_radar/dsp/decimation.py` | Propriété `stages` (pour l'audit), sortie complex128 optionnelle |
| M2 | `iot_radar/dsp/phase.py` | `fit_circle_pratt`, `track_center` (centres glissants, interpolation), démodulation avec un centre variable |
| M3 | `iot_radar/dsp/contamination.py` (nouveau) | `ContaminationMeter` (indice par bloc, à état), `contamination_weight` |
| | `iot_radar/dsp/estimation.py` | `weighted_lomb_scargle`, `remove_segment_offsets` |
| | `iot_radar/dsp/detection.py` | Cause de la pause (`pause_reason`) |
| | `iot_radar/pipeline.py` | Anomalies, attribution, confiance pondérée, saturation |
| M4 | `iot_radar/measurement.py` (nouveau) | Réglages, `MeasurementResult`, `MeasurementStatus`, règle de décision |
| | `iot_radar/pipeline.py` | `start_measurement()`, accumulation, analyse finale, `PipelineOutput.measurement` |
| Configuration | `configs/radar.yaml` | Sections `motion_robustness` et `measurement` |
| Interface | `iot_radar/ui/dashboard.py` | Poids courant, part contaminée, état de la mesure, touche `m` |
| Scripts | `scripts/run_radar.py` | `--measure` |
| | `scripts/evaluate_motion_robustness.py` (nouveau) | Évaluation (§6) |
| Simulation | `iot_radar/acquisition/disturbers.py` (nouveau) | Marcheur, balancement, geste, masquage |
| | `iot_radar/acquisition/sources.py` | `CWSimulationSource`, avec perturbateurs |
| | `iot_radar/acquisition/recording.py` | `/ground_truth/disturbances` |

### 3.2 M1 — Filtrage Doppler (Crochiere & Rabiner, 1981)

- **Filtre :** FIR à phase linéaire (symétrique), fenêtre de Kaiser, à 100 Hz, appliqué avec les mêmes coefficients réels sur I et Q. Il est causal, à état d'un bloc à l'autre.
- **Gabarit :**
  - bande passante jusqu'à 1,0 Hz, ondulation < 0,01 dB (fenêtre de Kaiser à 60 dB), ce qui laisse passer les harmoniques respiratoires jusqu'à 1 Hz ;
  - coupure à 1,5 Hz (−6 dB) ;
  - bande atténuée dès **2,0 Hz** avec **≥ 60 dB**.
  - La largeur de transition (1 Hz) est configurable.
- **Taille :** loi de Kaiser, N ≈ (60 − 7,95)/(2,285 · 2π · 1,0/100) ≈ **363 coefficients**.
- **Retard :** (N − 1)/2 = 181 échantillons à 100 Hz, soit **1,81 s**, documenté dans la docstring et dans le README.
- **Décimation ×5 vers 20 Hz :** les bandes repliées (19–21 Hz, 39–41 Hz) sont dans la bande atténuée du FIR. L'exigence de 60 dB est donc tenue par construction ; l'audit le vérifie.
- **Effets :**
  - la latence de la décision augmente de 1,8 s (contre 0,12 s pour le Tchebychev actuel) ;
  - après une discontinuité, le FIR met 3,6 s à se remplir : le temps de chauffe passe de 1 s à `max(1 s, 3,6 s)`.
- **Désactivé :** dernier étage Tchebychev ×5 actuel, donc ancien comportement.

### 3.3 M2 — Suivi glissant du centre de l'arc

Références : Park, Boric-Lubecke et Lubecke (2007), Zakrzewski et al. (2012).

- **Principe.** Un perturbateur modifie les réflexions fixes : le centre C(t) dérive. On l'estime sur des **sous-fenêtres de 10 s** (2 à 3 cycles respiratoires) **tous les 1 s**, à l'intérieur de la fenêtre d'analyse.
- **Continuité.** C(t) est **interpolé linéairement** entre les instants centraux des sous-fenêtres, et maintenu constant aux bords. La phase vaut ensuite φ(t) = unwrap(arg(z(t) − C(t))).
  - Comme C(t) est continu, z − C l'est aussi, et la phase aussi : **aucun saut artificiel** n'est introduit lors des mises à jour.
  - Une sous-fenêtre non valide (arc < 0,3 rad, résidu > 0,3) est ignorée et l'interpolation passe par-dessus. Si aucune n'est valide, on se replie sur le traitement actuel.
- **Comparaison des estimateurs** (méthodes décrites par Zakrzewski et al.). Mesure : 300 tirages, sous-fenêtre de 10 s à 20 Hz, 2,45 GHz, centre à 30 fois le rayon. On donne l'erreur médiane sur le déplacement retrouvé, puis l'erreur sur le centre relative au rayon.

| p-p | Arc | SNR | Moyenne | Kåsa | **Pratt** | Géométrique (Kåsa + Gauss-Newton) |
|---|---|---|---|---|---|---|
| 2 mm | 0,21 rad | 20 dB | 59 mm / 1,00 r | 62 mm / 1,00 r | 58 mm / 1,00 r | 56 mm / 1,00 r |
| 2 mm | 0,21 rad | 30 dB | 33 mm / 1,00 r | 28 mm / 0,99 r | 0,70 mm / 0,61 r | 0,70 mm / 0,60 r |
| 5 mm | 0,51 rad | 20 dB | 37 mm / 0,98 r | 31 mm / 0,96 r | **1,82 mm** / 0,55 r | 1,31 mm / 0,45 r |
| 5 mm | 0,51 rad | 30 dB | 21 mm / 0,98 r | 4,9 mm / 0,77 r | **0,28 mm** / 0,10 r | 0,27 mm / 0,10 r |
| 10 mm | 1,03 rad | 20 dB | 21 mm / 0,94 r | 5,2 mm / 0,66 r | **0,76 mm** / 0,09 r | 0,75 mm / 0,08 r |
| 10 mm | 1,03 rad | 30 dB | 11 mm / 0,94 r | 0,78 mm / 0,18 r | **0,23 mm** / 0,02 r | 0,23 mm / 0,02 r |
| Temps de calcul | | | 7 µs | 110 µs | **120 µs** | 370–850 µs |

**Choix : Pratt.**
- Il est aussi précis que l'ajustement géométrique dès que l'arc dépasse ~0,5 rad.
- Il est **non itératif** : pas de convergence à surveiller, et 3 à 7 fois plus rapide. Cela compte ici : 11 sous-fenêtres par mise à jour, deux mises à jour par seconde.
- Kåsa est fortement biaisé sur les arcs courts, comme la moyenne : il est écarté.
- L'ajustement géométrique reste celui de la fenêtre entière (M2 désactivé).

En dessous de ~0,3 rad, aucune méthode ne tient. Le contrôle de validité existant s'applique alors, avec repli sur la démodulation linéaire.

**Dérive à l'intérieur d'une sous-fenêtre (constaté à l'étape 3).**
- **Le problème :** si le centre se déplace pendant une sous-fenêtre d'autant que l'arc respiratoire, les points quittent le cercle. Avec un ajustement statique, quelle que soit la méthode (Pratt, géométrique, ou même un ajustement à centre mobile initialisé par Pratt), l'erreur atteint des dizaines de millimètres.
- **La solution :** retirer d'abord la **tendance linéaire de chaque sous-fenêtre**. On calcule la pente complexe de z(t) par moindres carrés ; l'oscillation respiratoire se moyenne sur 2 à 3 cycles. Pratt est ensuite appliqué aux points corrigés, ce qui reste non itératif.

Mesure : médiane sur 30 tirages, 8 mm de respiration, sous-fenêtres de 10 s ; erreur sur le déplacement à 20 dB / 30 dB.

| Dérive du centre sur 20 s | Pratt seul | Pratt après retrait de la tendance |
|---|---|---|
| 0 | 0,86 / 0,25 mm | 0,82 / 0,26 mm |
| 0,5 rayon | 1,14 / 0,75 mm | 0,90 / 0,45 mm |
| 1 rayon | 15,5 / 28,9 mm | 1,14 / 0,79 mm |
| 2 rayons | 14,6 / 13,6 mm | 1,77 / 1,55 mm |
| 4 rayons | — | 3,5 / 3,3 mm |

- Des sous-fenêtres de 6 s suivent mieux une forte dérive (1,7 / 1,4 mm à 4 rayons), mais sont plus bruitées sans dérive. La valeur par défaut reste 10 s, configurable.
- Une variation brusque et ample du centre (masquage d'un mur par un passage) reste hors de portée de M2. Elle relève de M3 (pondération).

**Sélection du modèle (constaté à l'étape 4, scènes de validation avec M2 actif par défaut).**
- **Retrait de tendance :** sur 2 à 3 cycles, la pente emporte aussi une partie de l'arc respiratoire. L'amplitude était biaisée de −17 % (6,7 mm pour 8 mm à 0,3 Hz). L'ajustement après retrait de la tendance n'est gardé que si son résidu relatif est inférieur à 0,8 fois celui de l'ajustement direct. Ce rapport vaut 1,00 sans dérive, et 0,64 à 0,75 pour une dérive de 0,3 rayon.
- **Centre suivi ou fixe :** un centre qui bouge d'une sous-fenêtre à l'autre transforme l'IQ immobile d'une apnée en dérive lente de la phase. On obtenait ainsi de fausses respirations et une alerte d'apnée intermittente. `demodulate_tracked` garde donc un ajustement fixe sur toute la fenêtre, sauf si le centre suivi donne un résidu relatif inférieur à 0,8 fois celui du centre fixe, c'est-à-dire s'il y a une dérive. Dans une scène calme, la méthode reste `arctan`, comme avant ; avec une dérive, elle devient `arctan_tracked`.

### 3.4 M3 — Pondération et estimation tolérante aux trous (VanderPlas, 2018)

- **Indice par bloc de 1 s, calculé à 100 Hz AVANT M1 :**

  ```
  κ = 10·log10( (E_hors / B_hors) / (E_bande / B_bande) )   [dB]
  ```

  - E_hors est l'énergie de l'IQ filtré passe-haut à 2 Hz, sur [2 Hz ; 50 Hz] en valeur absolue (B_hors = 2·48 Hz).
  - E_bande est l'énergie de l'IQ filtré passe-bande sur [0,1 ; 1 Hz] en valeur absolue (B_bande = 2·0,9 Hz). Le continu, c'est-à-dire le statique, est exclu.
  - Les filtres sont des Butterworth d'ordre 4 à état, en continu, démarrés en régime établi sur le premier échantillon : sinon l'échelon du statique fait sonner le passe-bande pendant des dizaines de secondes.
  - **Normalisation par les largeurs de bande (Q2)** : un bruit blanc vaut 0 dB. Mesuré en simulation : une cible qui respire est entre −30 et −20 dB ; un marcheur à 3,5 m remonte κ jusqu'à −5 dB environ ; un geste à 1,5 m, autour de 0 dB.
  - Conséquence de la normalisation : une raie Doppler est « diluée » dans les 96 Hz hors bande. Le poids ne baisse donc que pour un perturbateur nettement plus fort que l'écho respiratoire (≈ 10 fois en amplitude pour w = 0,5). C'est voulu : derrière M1, un perturbateur hors bande plus faible est déjà supprimé linéairement.
- **Poids :** fonction logistique lisse, `w = 1 / (1 + exp((κ − κ₀)/s))`, avec κ₀ = 9 dB et s = 2 dB (κ₀ valait 6 dB jusqu'à l'évaluation de l'étape 5, voir §4).
  - On obtient w ≈ 0,99 à 0 dB, 0,5 à 9 dB et 0,05 à 15 dB.
  - Un échantillon de poids inférieur à 0,5 compte comme « contaminé ».
  - Les poids sont retardés du retard de M1 (1,82 s) pour rester alignés avec le temps lent filtré.
  - Un bloc qui contient des échantillons saturés reçoit le poids 0.
- **Activité hors bande** (ajoutée à l'étape 4) : densité dans la bande du Doppler humain [3 ; 28 Hz] rapportée à la densité dans une bande au-dessus, [30 ; 38 Hz], qui ne contient que le bruit du récepteur et reste dans la bande passante de la décimation. Elle dit « quelqu'un bouge autour du radar », quelle que soit la force de la cible et quel que soit le passé. Une première version comparait la densité hors bande à son plancher sur la dernière minute : un perturbateur présent depuis le début devenait le plancher, et ses anomalies étaient prises pour la cible (11 % de pause dans une pièce vide avec gestes). La bande commence à 3 Hz pour que les bandes latérales d'une respiration profonde (excursion de 3 rad) restent dehors. Bruit seul : 0 ± 3 dB par bloc ; marcheur à 3–6 m : +8 à +22 dB. Seuil : 6 dB. Le premier bloc après un redémarrage vaut 0 dB, car les filtres sont en régime transitoire.
- **Estimation pondérée :** périodogramme de **Lomb-Scargle pondéré**, sous la forme généralisée avec moyenne flottante (VanderPlas, §5–6), sur le déplacement de la fenêtre avec les poids des échantillons.
  - Pour chaque fréquence, on fait un ajustement pondéré de A·cos + B·sin + c ; la puissance est la réduction du χ² pondéré.
  - Les échantillons de poids nul ne comptent pas : l'estimation tolère les trous.
  - Chaque segment entre deux trous reçoit son propre décalage (`remove_segment_offsets`) : un passage peut faire glisser la phase de 2π, soit une marche de λ/2 = 61 mm (voir ci-dessous).
  - Les métriques existantes (fréquence, SNR, concentration) sont calculées sur ce spectre, sans changer de code (`breathing_metrics`).
- **Confiance :** confiance instantanée × poids moyen de la fenêtre. Elle diminue donc avec la part de segments contaminés.
- **Anomalies dans la bande** (constat de l'étape 4) : un marcheur qui coupe le faisceau passe par une vitesse radiale nulle. Pendant ~1 s, son écho est dans la bande respiratoire, et M1 ne peut rien. S'il est d'amplitude comparable à la cible, le phaseur autour du centre fait un tour : la phase glisse de 2π. Vu de la sortie, c'est le même signal qu'un mouvement de la cible (même rayon d'arc écrasé par M1, même énergie hors bande pendant l'événement). Ce qui les distingue, c'est l'avant : un passant n'atteint la bande qu'en ralentissant à travers les Doppler hors bande, on le voit venir ; la cible qui bouge démarre dans la bande.
  - **Anomalie** : suite d'échantillons dont le crête à crête local (3 s centrées, sur le déplacement non redressé : une tendance globale ajustée à travers une marche inclinerait chaque segment) dépasse `motion_ptp_mm`. Poids 0.
  - **Excursion** : crête à crête local supérieur à `excursion_factor` (2,5) × sa médiane sur la fenêtre et, loin de toute activité hors bande, à `excursion_floor_mm` (12 mm, sans quoi une apnée qui remplit la fenêtre ferait couper la respiration restante ; près d'un passant, ses bords de 1 à 3 mm seraient au contraire gardés), même sous `motion_ptp_mm` (par exemple, le transitoire du centre suivi quand un écho statique saute, comme une main qui se met à bouger). Poids 0.
  - **Segments courts** : un segment de moins de `min_segment_s` (5 s) entre deux coupures reçoit le poids 0. Avec son propre décalage, il ne peut pas contenir une respiration, et sa dérive donnait un faux pic au bord de la bande (pièce vide traversée par des marcheurs, étape 5).
  - **Attribution** : une anomalie est attribuée à un passant si l'activité hors bande a dépassé son seuil dans les `activity_lead_s` (10 s) qui précèdent son début. Sinon, elle est attribuée à la cible.
  - **Deuxième passe (M2 avec M3)** : les sous-fenêtres de M2 qui contiennent une anomalie donnent un mauvais centre, car les points quittent le cercle. Il en résulte de fausses marches jusqu'à une demi-sous-fenêtre (5 s) de l'anomalie. Le centre est donc réajusté sans les échantillons coupés (`track_center(..., exclude=...)`), puis l'analyse est refaite. Cette seconde passe n'est gardée que si elle coupe moins d'échantillons. Après un saut de l'écho statique, comme une main qui se met à bouger, ce n'est pas le cas : aucune interpolation linéaire du centre ne le suit.
  - Constat en simulation : le glissement de 2π du marcheur venait surtout du centre faussé. Avec la seconde passe, la fenêtre redevient souvent entièrement valide, et la respiration est détectée 95 % du temps, contre 81 % sans elle.
- **Pause** (M3 activé) réservée à deux cas :
  - **saturation de l'ADC** (au moins un échantillon d'un bloc à ±2047), maintenue `motion_hold_s` ; `pause_reason = "saturation"` ;
  - **anomalie de la cible** en cours (dans les `motion_window_s` dernières secondes) ; `pause_reason = "motion"`.
  - L'état `MOTION` garde son nom.
  - M3 désactivé : règle actuelle (crête à crête de la fenêtre et des 5 dernières secondes).
- **Limites connues :**
  - si la cible bouge pendant qu'un passant est actif, l'anomalie est attribuée au passant : pas de pause, mais poids 0 sur l'anomalie ;
  - un perturbateur qui démarre d'un coup dans la bande, sans phase hors bande préalable, est pris pour la cible. C'est le cas d'une personne qui se met à marcher depuis l'arrêt près de la cible : son Doppler monte depuis 0 à travers la bande. On obtient une courte pause, 5 % de 90 s en simulation, pour un départ à 0,5 m/s à 5 m.

### 3.5 M4 — Mode « mesure »

- **Déclenchement par l'opérateur :**
  - touche `m` du tableau de bord ;
  - `scripts/run_radar.py --measure` en mode sans affichage (une mesure, puis sortie) ;
  - `VitalSignsPipeline.start_measurement(duration_s=None)` dans le code.
  - L'interface ne touche pas au pipeline : elle lève un drapeau que la boucle de production lit avant chaque bloc.
- **Pendant la mesure** (30 s par défaut), la chaîne continue tourne normalement. La mesure accumule le temps lent, ses poids et l'activité hors bande. Elle part des échantillons issus de l'IQ reçu après le déclenchement : ceux déjà dans le filtre de M1 (1,82 s) sont sautés, si bien que la fenêtre couvre exactement le silence. L'état publié est `running`, avec le temps écoulé.
- **À la fin, analyse unique de la fenêtre de mesure**, avec les mêmes traitements M1 à M3, deuxième passe comprise :
  - **décision** (`measurement_decision`) :
    - `breathing` si la confiance pondérée ≥ `threshold_on` ;
    - `no_breathing` si elle est < `threshold_off` ;
    - `inconclusive` sinon (`uncertain`), ou, quelle que soit la confiance, en cas de saturation (`saturation`), de mouvement de la cible (`target_motion` : une anomalie attribuée à la cible), de discontinuité du flux (`discontinuity`, la mesure s'arrête aussitôt) ou de part valide (poids ≥ 0,5) < `min_valid_fraction` (`too_contaminated`, 0,5) ;
  - **fréquence respiratoire** (Lomb-Scargle pondéré), si la décision est `breathing` ; **confiance** ;
  - **résumé de qualité** : part contaminée, part saturée, poids moyen, part avec activité autour du radar, mouvement de la cible, discontinuité.
- **Structures** (`iot_radar/measurement.py`) : `MeasurementSettings`, `MeasurementResult`, `MeasurementStatus` (`idle`, `running`, `done`) dans `PipelineOutput.measurement`. Le dernier résultat reste dans les sorties suivantes, pour l'affichage.
- **Mode continu :** inchangé et toujours disponible. La mesure s'y superpose.

---

## 4. Paramètres par défaut

```yaml
motion_robustness:
  doppler_rate_hz: 100.0      # D1 : Nyquist 50 Hz > marche à 1,5 m/s (24,5 Hz à 2,45 GHz ; 35 Hz à 3,5 GHz)
  doppler_filter:             # M1
    enabled: true
    cutoff_hz: 1.5            # demande ; garde les harmoniques respiratoires ≤ 1 Hz
    transition_hz: 1.0        # bande passante ≤ 1,0 Hz, atténuée ≥ 2,0 Hz
    stopband_attenuation_db: 60.0
    noise_band_hz: [0.1, 1.4] # plancher de bruit du détecteur, dans la bande passante
  alias_check:
    band_hz: 1.0              # f_b
    min_attenuation_db: 60.0  # exigence de la demande, vérifiée au démarrage
  center_tracking:            # M2
    enabled: false            # désactivé par défaut après l'évaluation (D2)
    window_s: 10.0            # 2 à 3 cycles à 12–18 /min
    step_s: 1.0
    method: pratt             # pratt | geometric | kasa (§3.3)
  contamination_weighting:    # M3
    enabled: true
    block_s: 1.0
    out_of_band_start_hz: 2.0 # = bord de la bande atténuée de M1
    in_band_hz: [0.1, 1.0]
    weight_midpoint_db: 9.0   # κ₀ (6 dB au départ ; 9 dB après l'évaluation)
    weight_slope_db: 2.0      # s
    contaminated_below: 0.5
    activity_band_hz: [3.0, 28.0]       # Doppler humain ...
    activity_reference_hz: [30.0, 38.0] # ... contre le bruit au-dessus
    activity_threshold_db: 6.0  # activité hors bande au-dessus du bruit
    activity_lead_s: 10.0       # fenêtre d'attribution d'une anomalie à un passant
    excursion_factor: 2.5       # excursions coupées au-delà de 2,5 × la médiane locale ...
    excursion_floor_mm: 12.0    # ... et de 12 mm (garde la respiration à côté d'une apnée)
    pause_on_saturation: true
measurement:                  # M4
  duration_s: 30.0
  min_valid_fraction: 0.5
```

Justifications :
- **100 Hz** : voir D1. 100 = 2²·5², et 1 MS/s / 100 Hz = 10 000 en facteurs ≤ 13.
- **Coupure à 1,5 Hz** : demande ; les harmoniques respiratoires montent jusqu'à ~1 Hz.
- **Atténuation dès 2 Hz** : en dessous des gestes (≥ 3 Hz) ; c'est aussi le début de la bande « hors bande » de M3.
- **10 s / 1 s** : 2 à 3 cycles respiratoires (demande) ; à 1 s, la dérive du centre due à un passage est suivie.
- **κ₀ = 9 dB** : 6 dB au départ, soit 6 dB au-dessus d'un bruit blanc, de sorte qu'une cible faible sans perturbateur garde un poids d'environ 0,95. Le balayage de l'étape 5 (κ₀ = 0, 3, 6, 9, 12 dB, M1+M2+M3, scènes S2, S5, S6, S7) donne pour 9 dB : Pd de 82 % au lieu de 70 %, fenêtres non conclues de 5 % au lieu de 18 %, sans fausse alarme ni pause en plus. 12 dB n'apporte rien de plus. À 6 dB, des gestes à 1,5 m rendaient non concluante une mesure sur quatre, alors que M1 les supprime déjà.
- κ₀ et s seront ajustés à l'étape 5 si l'évaluation le justifie ; tout changement sera documenté dans `MOTION_ROBUSTNESS_EVAL.md`.

---

## 5. Simulation des perturbateurs

Modèle simplifié d'après Chen et al. (IEEE TAES, 2006).

- **Géométrie 2D en SISO :** le radar est à l'origine, axe de visée +y ; la cible est en (0, `target_range_m`). Son modèle de phase reste inchangé.
- **Marcheur :**
  - **Corps :** tronc, deux jambes et deux bras, comme diffuseurs ponctuels (SER relatives 1 ; 0,25 ; 0,25 ; 0,1 ; 0,1).
  - **Mouvement :** le tronc suit une trajectoire rectiligne (départ, direction) à vitesse constante, entre 0,5 et 1,5 m/s. Jambes et bras oscillent le long de la marche, de ±0,3 m et ±0,15 m, en opposition de phase. La fréquence de pas vaut v / 1,4 m, soit 0,36 à 1,07 Hz.
  - **Amplitude** ∝ √SER / R² : on donne l'amplitude à 1 m.
  - **Phase** = −4π·R_i(t)/λ pour chaque diffuseur.
- **Balancement :** diffuseur oscillant radialement, 3 cm à 0,8 Hz.
- **Hors de son intervalle [start_s ; stop_s[, un perturbateur est immobile** : avant, à sa position de départ ; après, là où il s'est arrêté. Son écho fait alors partie du statique. Un écho allumé d'un coup serait un échelon, qui tombe lui-même dans la bande respiratoire, et personne n'apparaît de nulle part. Le masquage s'applique aussi à une personne immobile.
- **Geste :** main oscillant de 8 cm à 2 Hz. La vitesse maximale est alors de 1 m/s, d'où un Doppler jusqu'à 16 Hz à 2,45 GHz.
- **Masquage et modulation du clutter :**
  - des réflecteurs statiques (murs) sont placés dans la scène ;
  - quand le tronc d'un perturbateur passe à moins de 0,25 m du segment radar → point (cible ou réflecteur), devant ce point, l'amplitude du trajet est atténuée jusqu'à −20 dB, avec une transition en cosinus surélevé ;
  - le centre C(t) dérive donc comme sur le terrain, et la cible peut être masquée.
- **Vérité terrain :**
  - `disturbance_intervals()` donne les intervalles où chaque perturbateur est actif : dans le champ (±60°, < 15 m) pour le marcheur, période configurée pour le balancement et le geste ;
  - ils sont enregistrés dans `/ground_truth/disturbances` (D3).
- **Sans perturbateur, la source reste identique au bit près**, car les perturbateurs ne consomment pas de tirage aléatoire.

---

## 6. Protocole d'évaluation

`scripts/evaluate_motion_robustness.py` ; résultats et figures dans `MOTION_ROBUSTNESS_EVAL.md`.

**Configurations comparées** : référence (M1, M2, M3 désactivés, pause actuelle), M1, M2, M3, M1+M2, M1+M3, M2+M3, M1+M2+M3.

**Scénarios simulés** :
- **Paramètres communs :** 2,45 GHz ou selon Q1 ; débit réduit à 20 kS/s pour la vitesse, avec une décimation identique à partir de 100 Hz ; 5 graines × 150 s.

| # | Scène | Rôle |
|---|---|---|
| S1 | Cible qui respire, seule | Contrôle |
| S2 | Cible + marcheur traversant à 3 m, 0,5 à 1,5 m/s | Cas de la demande |
| S3 | Cible + marcheur à 8 m | Activité lointaine |
| S4 | Cible + balancement d'un témoin à 2 m | Mouvement dans la bande |
| S5 | Cible + gestes à 1,5 m | |
| S6 | Pièce vide + marcheur | Fausses alarmes |
| S7 | Pièce vide + gestes | Fausses alarmes |
| S8 | Cible qui bouge elle-même | La pause doit rester |

**Enregistrements** : les 69 anciennes sessions (`breathing` / `empty`) et, si possible, ceux de Q4.

**Mesures :**
- **Part du temps en pause :** sorties `MOTION` de la chaîne continue, après le chauffage.
- **Pd et Pfa sur des fenêtres disjointes de 30 s :** décision du mode mesure (§3.5) sur chaque fenêtre.
  - Pd = P(`breathing` | cible présente) ; Pfa = P(`breathing` | pas de cible).
  - On rapporte aussi la part `inconclusive`.
- **Erreur de fréquence :** |estimée − vraie| en respirations/min, sur les fenêtres décidées `breathing` (simulation seulement).

**Figures :**
- part de pause par configuration et par scénario ;
- Pd/Pfa par configuration ;
- erreur de fréquence ;
- un exemple temporel de S2 : IQ, poids, état, sans et avec M1+M2+M3.

---

## 7. Hors périmètre : annulation spatiale et mouvement de la cible

### 7.1 Annulation spatiale par formation de voies (MVDR, Van Veen & Buckley, 1988)

**Principe.** Avec le réseau virtuel MIMO de M = 4 éléments (`LOCALIZATION_PLAN.md`), un filtre spatial **w = R⁻¹·a(θ_c) / (a(θ_c)ᴴ·R⁻¹·a(θ_c))** garde la direction de la cible θ_c et place des zéros dans les directions des perturbateurs. On peut annuler au plus M − 1 = 3 perturbateurs.

**Ce qu'il apporte** que M1 à M3 ne peuvent pas apporter : la suppression des perturbateurs **dans la bande** respiratoire.
- un témoin qui se balance à 0,8 Hz ;
- la modulation du clutter par un passage ;
- un autre souffle.

En SISO, ces composantes sont spectralement indiscernables de la cible.

**Covariance sans annulation du signal.** Le défaut classique du MVDR est d'annuler la cible quand elle contamine la covariance. Ici, on peut estimer **R sur la partie hors bande** (2 à 50 Hz de l'IQ à 100 Hz), où il n'y a que les perturbateurs et le bruit, puis appliquer w à la partie dans la bande. Cette discrimination Doppler permet d'estimer la covariance des interférences sans la cible.

**Limites :**
- un perturbateur purement dans la bande (balancement sans composante hors bande) n'est pas vu par R hors bande ; il faut alors une covariance dans la bande, avec un risque d'annuler la cible ;
- 3 degrés de liberté seulement ;
- sensibilité aux erreurs de calibration ; une charge diagonale est nécessaire ;
- la direction de la cible doit être connue (la DoA de la localisation).

**Ce que la conception actuelle prépare :**
- un flux à **100 Hz** avant M1, où un traitement spatial linéaire s'insère ;
- un **indice de contamination** par bloc, qui indiquera quand estimer R, puis par voie ;
- des traitements M1 et M3 **linéaires en IQ et par voie**, applicables à chaque voie virtuelle ;
- des sorties sous forme de structures de résultats.

**À modifier :** fusionner avec `feature/mimo-localization`, puis insérer le MVDR entre Decimator A et M1, voie par voie.

### 7.2 Compensation du mouvement de la cible elle-même

- **Le problème :** un mouvement de la victime (déplacement, changement de posture) est dans la bande respiratoire, et souvent bien plus ample qu'elle. Ni M1 ni M3 ne peuvent le séparer de la respiration.
- **Ce qu'on fait :** la pause reste la bonne réponse (§3.4). M2 aide à repartir vite, puisqu'il ré-estime le centre après le mouvement.
- **Pistes, non implémentées :**
  - segmentation et rejet des segments en mouvement, avec reprise de l'estimation sur les segments stables (le Lomb-Scargle pondéré le permet déjà en donnant un poids nul) ;
  - modèle respiration + tendance lente, par régression robuste ;
  - en MIMO, angle et distance pour suivre la cible.

---

## 8. Risques

| Risque | Conséquence | Mitigation |
|---|---|---|
| Balancement ou mouvement d'un témoin **dans la bande** (0,8 Hz) | Non vu par M3 (pas d'énergie hors bande), pas retiré par M1 | La fréquence est cherchée dans 0,1–0,5 Hz ; limite documentée ; MVDR plus tard (§7.1) |
| Cible faible (énergie dans la bande proche du bruit) | κ ≈ 0 dB, poids ≈ 0,95 : sans effet ; mais une cible faible et un perturbateur modéré donnent des poids bas | κ₀ ajusté par l'évaluation ; part valide minimale dans M4 |
| Retard de M1 (1,8 s) et chauffe de 3,6 s après une discontinuité | Décision plus tardive | Documenté ; 30 s de mesure ≫ 3,6 s |
| Sous-fenêtres de M2 courtes (10 s) | Centre plus bruité qu'avec 20 s | Pratt ≈ géométrique (§3.3) ; repli si non valide |
| Peu d'échantillons de poids élevé | Pics parasites du Lomb-Scargle | Confiance × poids moyen ; `min_valid_fraction` |
| Modèle de marche simplifié | Évaluation optimiste | Paramètres variés ; données réelles demandées (Q4) |
| Anciennes sessions déjà filtrées passe-haut à 2 kHz, sans annotation de perturbation | Évaluation réelle limitée | Signalé dans le rapport |
| Fusion future avec `feature/mimo-localization` (`pipeline.py`, `sources.py`, `radar.yaml`) | Conflits | Ignorée pour le moment (Q3) ; modifications localisées |
| 2,45 GHz est aussi la fréquence des fours à micro-ondes | Brouillage possible près d'une cuisine | Documenté ; la porteuse reste configurable |

---

## 9. Étapes et commits

**Étape 1**
- `docs: add MOTION_ROBUSTNESS_PLAN.md`, puis les décisions Q1 à Q4
- `config: SISO carrier at 2.45 GHz` (Q1)

**Étape 2 — Simulation des perturbateurs**
- `test: pin the single-channel simulation and phase pipeline outputs` (Q3) ;
- `feat(sources): walking person, sway and gesture disturbers with path masking` ;
- `feat(recording): disturbance intervals in the ground truth`.

**Étape 3 — M1 et M2**
- `feat(dsp): Doppler low-pass and aliasing audit (M1)` ;
- `feat(pipeline): Doppler-rate stage and M1 in the chain` ;
- `feat(dsp): Pratt fit and sliding centre tracking (M2)` ;
- `feat(pipeline): M2 in the window analysis`.
- **Tests :**
  - 19 Hz et toute fréquence repliée dans ±1 Hz atténués de ≥ 60 dB ;
  - respiration retrouvée (amplitude et fréquence) ;
  - centre qui dérive : phase continue, déplacement retrouvé.

**Étape 4 — M3 et M4**
- `feat(dsp): contamination index and weights (M3)` ;
- `feat(dsp): weighted Lomb-Scargle periodogram` ;
- `feat(pipeline): weighted estimation and the new pause rule` ;
- `feat(pipeline): measurement mode (M4)` ;
- `feat(ui): weight, contaminated share and measurement in the dashboard`.
- **Tests :**
  - un marcheur à 3 m ne déclenche ni pause ni fausse détection ;
  - un mouvement de la cible déclenche toujours une pause ;
  - le mode mesure rend une décision et un résumé de qualité.

**Étape 5**
- `feat(scripts): evaluate_motion_robustness.py` ;
- `docs: MOTION_ROBUSTNESS_EVAL.md`.

**Étape 6**
- `docs: motion robustness in README.md and README.fr.md`, puis résumé final.

Chaque commit concerne une modification cohérente, avec la suite verte. Si un test change parce que le comportement change volontairement, le message de commit l'explique. Aucune fusion sans votre accord.

**Écarts constatés pendant la réalisation** (détails dans les messages de commit) :
- **Simulation :** les perturbateurs restent immobiles hors de leur intervalle (un écho allumé d'un coup tombait dans la bande). L'amplitude par défaut d'un tronc est 12,7 à 1 m, et non 500.
- **M1 :** le plancher de bruit du détecteur est mesuré dans la bande passante, car M1 vide la bande de référence.
- **M2 :**
  - le centre suivi n'est utilisé que s'il s'ajuste mieux qu'un centre fixe ;
  - le retrait de tendance n'est gardé que s'il réduit le résidu, sans quoi l'amplitude était biaisée et l'alerte d'apnée devenait intermittente ;
  - une seconde passe réajuste le centre sans les anomalies.
- **M3 :**
  - anomalies et excursions coupées, avec un plancher de 12 mm ;
  - attribution à un passant par l'activité hors bande, mesurée contre le bruit au-dessus du Doppler humain ;
  - un décalage par segment entre les trous.
- **Tests de validation :** l'alerte d'apnée arrive 1,82 s plus tard et le préchauffage dure 3,6 s, à cause du retard de M1.

### Références

- R. E. Crochiere, L. R. Rabiner, « Interpolation and Decimation of Digital Signals — A Tutorial Review », Proc. IEEE 69(3), 1981.
- B.-K. Park, O. Boric-Lubecke, V. M. Lubecke, « Arctangent Demodulation With DC Offset Compensation in Quadrature Doppler Radar Receiver Systems », IEEE T-MTT 55(5), 2007.
- M. Zakrzewski, H. Raittinen, J. Vanhala, « Comparison of Center Estimation Algorithms for Heart and Respiration Monitoring With Microwave Doppler Radar », IEEE Sensors J. 12(3), 2012.
- J. T. VanderPlas, « Understanding the Lomb–Scargle Periodogram », ApJS 236(1), 2018.
- V. C. Chen, F. Li, S.-S. Ho, H. Wechsler, « Micro-Doppler Effect in Radar: Phenomenon, Model, and Simulation Study », IEEE TAES 42(1), 2006.
- B. D. Van Veen, K. M. Buckley, « Beamforming: A Versatile Approach to Spatial Filtering », IEEE ASSP Magazine 5(2), 1988.
