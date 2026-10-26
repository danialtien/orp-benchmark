# Original launchers (provenance only)

`run_revision.sh` is the script that launched D+10 (`ctrl`), IRRA+A (`irra`), D+A seeds 1
and 2 (`seed 1`, `seed 2`) and D+A′ (`disjoint`), and scored them. It is kept verbatim,
with its original relative paths (`../Dataset/Result/...`). Its `RECIPE` array is the
D+A recipe. Its parity gate diffed every new log header against the D+A (seed 0) log
and killed the run on any unintended difference.

Use `tools/run_config.py` with `configs/train/*.yaml` to reproduce runs: the configs
list the same arguments, plus the defaults spelled out. The launchers of D
(`train_isatmp.sh D cuhk_author sdm 0`) and of the aerial Stage 2
(`train_docc_nomad.sh full`) were internal scripts with machine-specific paths. Their
complete argument lists are in `configs/train/D.yaml` and
`configs/train/DA_nomad_s*.yaml`.
