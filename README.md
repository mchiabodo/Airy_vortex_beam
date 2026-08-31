# Faisceau d'Airy 2D + Vortex optique — Simulation numérique

Simulation de la **propagation libre d'un faisceau d'Airy 2D portant un vortex
optique** (moment angulaire orbital, OAM), en régime paraxial, par **méthode
split-step de Fourier** (Beam Propagation Method).

Projet de recherche : correction et validation d'un code de propagation existant,
puis établissement d'une **loi analytique de la dérive du centre de masse** du
faisceau et sa vérification numérique.

## Aperçu physique

Le champ initial (à `z = 0`) est un faisceau d'Airy 2D tronqué multiplié par un
vortex optique de charge topologique `m`, centré en `(x_v, y_v)` :

```
A(x,y,0) = Ai(x/x₀)·Ai(y/x₀)·exp(a(x+y)/x₀) · ρ^|m|·exp(i m φ)
           └────────── Airy 2D tronqué ─────────┘   └── vortex OAM ──┘
```

Il est propagé via l'équation paraxiale (Schrödinger optique)

```
2 i k₀ ∂A/∂z + ∇⊥²A = 0
```

résolue **exactement dans l'espace de Fourier** à chaque pas :
`Â *= exp(-i K² dz')`.

## Résultats clés

- **Énergie conservée à 100 %** et trajectoire du centre de masse **rectiligne**
  (R² = 1,0000), conforme au théorème d'Ehrenfest — après correction d'un bug
  d'apodisation qui courbait artificiellement la trajectoire.
- **Loi analytique exacte** de la dérive :
  `⟨r⟩(z) = ⟨r⟩(0) + (⟨k⊥⟩/k₀)·z`, pente prédite sans propagation, avec un
  accord mesure/prédiction **< 0,5 %**.
- **Moment angulaire orbital** `⟨Lz⟩/ℏ ≈ m`, conservé en propagation libre
  (pente 0,995 ; R² = 0,999).

## Fonctionnalités du code

Le fichier de simulation contient **5 modes**, sélectionnés en bas du fichier via
la variable `mode` :

| Mode        | Ce qu'il produit                                              |
|-------------|--------------------------------------------------------------|
| `single`    | Intensité / phase pour une charge `m` + GIF                  |
| `charges`   | Trajectoire du centre de masse pour `m = 0..5`               |
| `loi`       | Loi de dérive (mesure vs prédiction) + cartographie          |
| `oam`       | Moment angulaire `⟨Lz⟩` (validation + conservation)          |
| `profil1D`  | Coupes 1D I(x), I(y) le long du lobe principal + GIF         |

> Le code et ses commentaires sont en **anglais** pour être réutilisables par tous.
> On choisit le mode via la variable `mode` tout en bas de `Airy.py`.

## Lancer

```bash
pip install numpy scipy matplotlib pillow
python Airy_free.py
```

## Structure du dépôt

| Dossier / fichier          | Contenu                                              |
|----------------------------|------------------------------------------------------|
| `Airy_free.py`             | Code de simulation, entièrement documenté            |
| `0_Revue_bibliographique/` | État de l'art des faisceaux d'Airy vortex            |
| `1_Rapport_de_synthese/`   | Rapport principal (10 p.) : corrections + loi + valid.|
| `2_Mode_d_emploi/`         | Notice d'utilisation du code (2 p.)                  |
| `3_Diapositives/`          | Diaporama de soutenance                              |
| `4_GIFs_soutenance/`       | Animations de propagation (voir note ci-dessous)     |
| `5_Etude_autoguerison/`    | Étude de l'auto-guérison de phase                    |

> **Note :** les animations GIF de démonstration sont volumineuses et ne sont pas
> versionnées dans ce dépôt (voir `.gitignore`). Elles sont disponibles sur
> demande. Les figures des rapports (`.png`) sont, elles, incluses.

## Méthode & implémentation

- Grille adimensionnée en unités de la largeur du lobe `x₀`, propagateur spectral
  `exp(-i K² dz')` (solution exacte de l'équation paraxiale).
- Fenêtre de Tukey appliquée **une seule fois** à l'injection (propagation libre
  sans perte) — point clé pour respecter la conservation d'énergie et le théorème
  d'Ehrenfest.
- Observables calculées par intégration spectrale : centre de masse `⟨r⟩(z)` et
  moment angulaire orbital `⟨Lz⟩(z)` (dérivées pseudo-spectrales).

## Technologies

Python 3 · NumPy · SciPy · Matplotlib · Pillow · LaTeX (rapports)
