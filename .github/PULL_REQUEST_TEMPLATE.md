<!-- SPDX-Symbol: GitHub pull request template; the repository's license (LICENSE) applies -->

## What and why

<!-- One paragraph: what changes, and the reason the change is wanted.
     If the CHANGELOG already carries an entry, its title is enough. -->

## Checks

- [ ] Source checks pass: `check_docs`, `check_headers`, `check_includes`,
      `check_tracked`, `check_raw_reads` (`ctest` runs them, or each script
      under scripts/ directly).
- [ ] `ctest -L validation` green in **both** modes, OMP=ON and OMP=OFF
      (the serial specialisations only compile without OpenMP).
- [ ] No `fixup!` / `squash!` / `amend!` subject lines: the CI refuses them.
- [ ] Every file a commit builds or references is in the commit
      (a registered target without its source breaks the configure).
- [ ] Documentation updated in **both** languages, and the names it puts in
      backticks exist in the tree (`check_docs` verifies both).
- [ ] A new build option is wired end to end: CMake option → generated
      `build_options.h` → `build_config` constant → a consumer.
