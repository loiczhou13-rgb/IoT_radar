# Localisation (phase 2) — placeholder

La phase 2 vise la **localisation** (distance/position) d’un survivant.

## Note importante

En **phase 1** (CW micro‑doppler), on détecte une respiration mais on **n’obtient pas de distance** fiable.  
Pour de la localisation, il faudra typiquement introduire une **modulation** (ex: FMCW) et une chaîne de traitement dédiée (range FFT, calibration, etc.).

## API prévue (à définir quand on démarre la phase 2)

- Entrées possibles :
  - paramètres RF (FC/FS/BW)
  - captures IQ synchronisées
  - méta‑données SDR (gains, configuration)
- Sorties possibles :
  - estimation(s) de distance + incertitude
  - indicateurs de qualité (SNR, cohérence)

