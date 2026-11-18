#!/bin/sh
# Copyright 2026 CEA (www.cea.fr)
# SPDX-License-Identifier: Apache-2.0
#
# apt_install.sh <package...>: apt-get update, then install, on a GitHub
# runner. Through archive.ubuntu.com rather than the runner's mirror list,
# whose Azure mirror stopped resolving for hours on 2026-11-11 ("Temporary
# failure resolving 'azure.archive.ubuntu.com'"); three attempts, 20 s apart.
set -u
sources=/etc/apt/sources.list.d/ubuntu.sources
if [ -f "$sources" ] && grep -q 'mirror+file:' "$sources"; then
    sudo sed -i 's#mirror+file:/etc/apt/apt-mirrors.txt#http://archive.ubuntu.com/ubuntu/#' "$sources"
fi
for attempt in 1 2 3; do
    if sudo apt-get update -q && sudo apt-get install -y -q "$@"; then
        exit 0
    fi
    echo "apt_install.sh: attempt $attempt failed" >&2
    sleep 20
done
exit 1
