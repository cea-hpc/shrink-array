# Alternatives, et quand les préférer

> English version: [alternatives.md](alternatives.md).

shrink-array remplace directement un `std::vector` que le programme continue de lire et
d'écrire en place, depuis des threads OpenMP : même espace d'indices, même `[]`, sans perte, et
toute valeur répétée est compressée chunk par chunk (`zero_any`), pas seulement zéro. D'autres
structures économisent la mémoire sur d'autres données, et y sont le meilleur choix :

| alternative | ce qu'elle fait | la préférer quand |
|---|---|---|
| formats creux (CSR / COO, SparseMatrix d'Eigen, `Mat` de PETSc) | stocke les entrées non nulles et leurs indices ; seul zéro est implicite | les données sont très creuses et le calcul est de l'algèbre linéaire creuse (SpMV, solveurs) ; la structure change rarement, car écrire hors d'elle oblige à la reconstruire |
| tables de hachage (`std::unordered_map`, tables à adressage ouvert) | une clé et une valeur par entrée stockée, quelques dizaines d'octets chacune, dispersées en mémoire | les entrées sont extrêmement peu nombreuses, sans structure spatiale, et leur étendue n'est pas connue à l'avance ; sous threads, il faut une table concurrente |
| compresseurs avec accès tableau : tableaux compressés de [zfp](https://github.com/LLNL/zfp), [Blosc2](https://www.blosc.org/) | compressent des blocs de données quelconques, valeurs variées comprises ; les tableaux à débit fixe de zfp sont avec perte et offrent `[]` à travers un cache de blocs ; Blosc2 est sans perte par défaut, par chunk et par bloc | les valeurs sont variées mais compressibles, ou une perte contrôlée est acceptable (zfp, SZ), ou le tableau est surtout écrit une fois puis lu, ou stocké (E/S, points de reprise) : une écriture au milieu décompresse puis recompresse un bloc |
| grilles 3D creuses : [OpenVDB](https://www.openvdb.org/) / NanoVDB, SPGrid, bibliothèques d'AMR | une hiérarchie de tuiles sur un domaine 3D, avec une valeur de fond et des tuiles constantes | les données sont un champ 3D manipulé comme une géométrie (bandes étroites, level sets, voisinages, opérations de topologie) plutôt que comme un tableau indexé par les boucles du code |

L'idée d'un chunk uniforme qui ne garde pas de buffer est proche des tuiles constantes
d'OpenVDB ; ce qu'ajoute shrink-array, c'est de rester un tableau plat : les boucles,
l'indexation et les écritures parallèles d'un code écrit pour `std::vector` restent telles
quelles. Là où chaque chunk contient des valeurs variées, aucune de ces structures n'aide
vraiment, sauf un compresseur, et shrink-array coûte un peu plus que le `std::vector`.
