#!/bin/bash
# refine around the best grid point, then validate the refined best on N=2000/3000 x 4 seeds
set -e
cd "$(dirname "$0")/.."                       # project root, wherever the project lives
.venv/bin/python scripts/n4_calibrate.py refine --procs 14 > results/logs/n4_refine.log 2>&1
read VSET B LC UNI < <(.venv/bin/python -c "
import json; R=json.load(open('results/n4_calibration.json'))
r=min([r for r in R.values() if r['n_wires']==2000 and r['network_seed']==1], key=lambda r:r['loss'])
print(r['V_set'], r['b'], r['lambda_crit'], int(r.get('unipolar', False)))")
EXTRA=""; [ "$UNI" = "1" ] && EXTRA="--unipolar"
.venv/bin/python scripts/n4_calibrate.py validate --V-set $VSET --b $B --lambda-crit $LC $EXTRA --procs 8 > results/logs/n4_validate.log 2>&1
echo "done V_set=$VSET b=$B lambda_crit=$LC unipolar=$UNI"
