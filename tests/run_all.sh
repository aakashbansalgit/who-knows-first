#!/usr/bin/env bash
# Run every stage's offline test on synthetic data (no network, no GPU). Each test prints ALL TESTS PASSED.
set -u
cd "$(dirname "$0")"
status=0
for t in test_*.py; do
  start=$(date +%s)
  if python3 "$t" > "/tmp/wkf_${t%.py}.log" 2>&1 && grep -q "ALL TESTS PASSED" "/tmp/wkf_${t%.py}.log"; then
    echo "PASS $t ($(( $(date +%s) - start ))s)"
  else
    echo "FAIL $t (log: /tmp/wkf_${t%.py}.log)"; status=1
  fi
done
exit $status
