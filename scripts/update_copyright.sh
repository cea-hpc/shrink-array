#!/bin/bash
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0

# The start year of the copyright lines to update
START_YEAR="2024"

# Every file git tracks (.h and .cpp)
git ls-files | grep -E '\.(h|cpp|inline\.h)$' | while read -r file; do
    
    # 1. The year of the file's last commit
    last_year=$(git log -1 --format=%ad --date=format:%Y -- "$file")
    
    # No commit yet: the current year
    if [ -z "$last_year" ]; then
        last_year=$(date +%Y)
    fi

    # 2. Does the file need a change? Look for "Copyright XXXX CEA"
    if grep -q "Copyright $START_YEAR CEA" "$file"; then
        
        if [ "$last_year" != "$START_YEAR" ]; then
            echo "Update: $file ($START_YEAR -> $START_YEAR-$last_year)"
            
            # "Copyright 2024 CEA" becomes "Copyright 2024-2026 CEA"
            sed -i "s/Copyright $START_YEAR CEA/Copyright $START_YEAR-$last_year CEA/g" "$file"
        fi

    elif grep -q "Copyright $START_YEAR-[0-9]\{4\} CEA" "$file"; then
        
        # Already a range (e.g. 2024-2025)
        current_range_end=$(grep -o "Copyright $START_YEAR-[0-9]\{4\}" "$file" | cut -d'-' -f2)
        
        if [ "$last_year" != "$current_range_end" ]; then
            echo "Update range: $file ($START_YEAR-$current_range_end -> $START_YEAR-$last_year)"
            sed -i "s/Copyright $START_YEAR-[0-9]\{4\} CEA/Copyright $START_YEAR-$last_year CEA/g" "$file"
        fi
    fi
done

echo "Done."
