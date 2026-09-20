#!/bin/bash

# Année de départ par défaut si non trouvée
START_YEAR="2024"

# Parcourir tous les fichiers suivis par git (.h et .cpp)
git ls-files | grep -E '\.(h|cpp|inline\.h)$' | while read -r file; do
    
    # 1. Récupérer l'année du dernier commit pour ce fichier
    last_year=$(git log -1 --format=%ad --date=format:%Y -- "$file")
    
    # Si le fichier n'a pas encore de commit, on prend l'année actuelle
    if [ -z "$last_year" ]; then
        last_year=$(date +%Y)
    fi

    # 2. Vérifier si on doit modifier le fichier
    # On cherche la ligne "Copyright XXXX CEA"
    if grep -q "Copyright $START_YEAR CEA" "$file"; then
        
        if [ "$last_year" != "$START_YEAR" ]; then
            echo "Mise à jour : $file ($START_YEAR -> $START_YEAR-$last_year)"
            
            # Remplacement de "Copyright 2024 CEA" par "Copyright 2024-2026 CEA"
            # On utilise -i.bak pour créer une sauvegarde temporaire (sécurité)
            sed -i "s/Copyright $START_YEAR CEA/Copyright $START_YEAR-$last_year CEA/g" "$file"
        fi

    elif grep -q "Copyright $START_YEAR-[0-9]\{4\} CEA" "$file"; then
        
        # Cas où il y a déjà un intervalle (ex: 2024-2025)
        current_range_end=$(grep -o "Copyright $START_YEAR-[0-9]\{4\}" "$file" | cut -d'-' -f2)
        
        if [ "$last_year" != "$current_range_end" ]; then
            echo "Mise à jour intervalle : $file ($START_YEAR-$current_range_end -> $START_YEAR-$last_year)"
            sed -i "s/Copyright $START_YEAR-[0-9]\{4\} CEA/Copyright $START_YEAR-$last_year CEA/g" "$file"
        fi
    fi
done

echo "Terminé."
