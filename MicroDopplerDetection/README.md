# Détection micro-Doppler (PlutoSDR)

## 1. Contexte physique

Le thorax d’une personne respirant se déplace périodiquement (quelques mm à ~0,2–0,5 Hz). En radar CW, ce mouvement module la phase du signal réfléchi (modulation de phase). En bande de base, l’onde prend la forme \(x(t) \approx e^{j\phi(t)}\). Le développement en séries de Bessel montre que le spectre contient des raies à \(0, \pm f_v, \pm 2f_v, \ldots\); l’amplitude de la raie utile à \(\pm f_v\) est liée à \(J_1(m)\) avec \(m = 4\pi D/\lambda\). Le clutter statique domine souvent à 0 Hz; le pipeline vise à révéler la raie de respiration malgré ce pic DC et le bruit.

## 2. Schéma du pipeline (modules Python)

```
[config.yaml]
     │
     ▼
┌─────────────────┐     pipeline/emission.py      Génération CW / CW+décalage
└────────┬────────┘
         ▼
┌─────────────────┐     pipeline/acquisition.py   TX cyclique + RX pyadi-iio (ou simulation)
└────────┬────────┘
         ▼
┌─────────────────┐     pipeline/decimation.py      Réduction f_s + garde anti-repliement
└────────┬────────┘
         ▼
┌─────────────────┐     pipeline/clutter.py         mean / IIR / MTI
└────────┬────────┘
         ▼
┌─────────────────┐     pipeline/windowing.py      Hann, Hamming, Blackman, none
└────────┬────────┘
         ▼
┌─────────────────┐     pipeline/spectrogramme.py  STFT → Z, S_dB, axes f et v
└────────┬────────┘
         ▼
┌─────────────────┐     pipeline/detection.py      Seuil SNR ou none
└────────┬────────┘
         ▼
┌─────────────────┐     utils/display.py           Waterfall matplotlib (dB, m/s)
└─────────────────┘
```

Orchestration : `main.py`.

## 3. Installation

```bash
cd /path/to/IoT_radar
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

- **pyadi-iio** : `pip install pyadi-iio` (déjà listé dans `requirements.txt` à la racine du dépôt).
- **libiio** (Linux) : `sudo apt install libiio-utils` — vérifier le périphérique avec `iio_info -s`.
- **Windows / macOS** : installer les pilotes / runtime Analog Devices pour PlutoSDR et libiio selon la documentation officielle.
- Autres paquets : `numpy`, `scipy`, `matplotlib`, `PyYAML` (voir `requirements.txt`).

## 4. Hack PlutoSDR (optionnel, AD9364 étendu)

Sur le Pluto (SSH, utilisateur `root`, mot de passe `analog` par défaut) :

```bash
fw_setenv compatible ad9364
pluto_reboot reset
```

Après redémarrage : plage indicative 70 MHz–6 GHz, bande passante IQ jusqu’à ~56 MHz (selon image firmware).

## 5. Utilisation

```bash
cd MicroDopplerDetection
python main.py --config config.yaml
python main.py --config config.yaml --simulation   # force la simulation (sans matériel)
```

Avec affichage graphique : backend interactif classique (`TkAgg`, `Qt5Agg`, etc.). Sans écran (Raspberry Pi, CI) : `MPLBACKEND=Agg` — la figure est fermée automatiquement sans fenêtre.

## 6. Paramètres de `config.yaml` (effet physique)

| Section | Paramètre | Rôle |
|--------|-----------|------|
| `logging` | `level` | Verbosité des journaux. |
| `sdr` | `uri` | Adresse Pluto (`ip:…` ou `usb:`). |
| `sdr` | `f_c` | Porteuse : fixe \(\lambda = c/f_c\) pour l’axe vitesse \(v = f\lambda/2\). |
| `sdr` | `f_s` | Cadence IQ : bande Doppler non décimée \(\pm f_s/2\) (théorique). |
| `sdr` | `rx_gain` / `tx_gain` | Compromis sensibilité / saturation RX. |
| `sdr` | `buffer_size` | Longueur d’un bloc `rx()`. |
| `sdr` | `n_frames` | Nombre de blocs concaténés : doit être suffisant pour que, après décimation, la durée couvre au moins une fenêtre STFT (`n_fft` échantillons à \(f_s/D\)). |
| `emission` | `mode`, `f_offset` | CW pur ou tonalité décalée pour éloigner l’énergie du DC numérique. |
| `decimation` | `D`, `f_max_utile` | Baisse \(f_s\); contrainte de Shannon \(f_s/D > 2 f_{\max}\). |
| `clutter` | `mode`, `alpha` | Atténuation du fouillis statique (moyenne, EMA, différence). |
| `windowing` | `mode` | Réduction des lobes de fuite spectrale sur chaque segment STFT. |
| `spectrogramme` | `n_fft`, `overlap` | Résolution fréquentielle \(\Delta f \approx f_{s,\mathrm{dec}}/n_{fft}\) et recouvrement temporel. |
| `detection` | bandes, `seuil_snr_dB` | Comparaison d’énergie respiration / bande de référence. |
| `affichage` | `dynamique_dB`, `ylim`, `colormap` | Lisibilité du waterfall. |
| `simulation` | `fv`, `D_mm`, `snr_dB` | Signal synthétique + clutter fort + bruit. |

## 7. Pièges courants

- **Saturation ADC** : échantillons « plaqués » vers \(\pm 2048\) (échelle entière) ; réduire `rx_gain` ou `tx_gain`.
- **Pic DC** : imperfections IQ, LO leakage ; moyenne globale, tonalité `cw_offset`, filtres adaptés.
- **Wi-Fi 2,4 GHz** : interférences ; mesurer le spectre à vide, changer de canal ou de bande si possible.
- **Bruit de phase** : limite la cohérence sur de longues acquisitions ; fenêtres plus courtes ou moyennage prudent.
- **Décimation et Shannon** : si \(f_s/D \le 2 f_{\max}\), le pipeline lève une exception explicite.
- **Isolation TX/RX** : croisement de polarisation, espacement des antennes (souvent \(\geq 30\) cm), atténuateurs.

## 8. Résultat attendu (waterfall)

Avec une respiration à \(f_v \approx 0{,}3\) Hz et une résolution suffisante après décimation / STFT, on observe des raies horizontales persistantes à des vitesses correspondant à \(\pm f_v\) (symétriques en Doppler), au-dessus du fond de clutter résiduel près de 0 Hz.

## Verrouillage des versions

À la racine du dépôt, `requirements.txt` liste les paquets minimaux pour ce pipeline. Pour figer tout le venv : `python -m pip freeze > requirements-lock.txt`.
