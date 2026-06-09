#!/bin/bash
DB="data/analysis.db"
prev=0
echo "Monitoring Phase 2... (Ctrl+C to stop)"
while true; do
  result=$(sqlite3 "$DB" "SELECT phase2_done, phase2_total FROM sessions WHERE id=15;")
  done=$(echo $result | cut -d'|' -f1)
  total=$(echo $result | cut -d'|' -f2)
  llm=$(sqlite3 "$DB" "SELECT count(*) FROM llm_results WHERE smell_commit_id IN (SELECT id FROM smell_commits WHERE session_id=15);")
  running=$(sqlite3 "$DB" "SELECT count(*) FROM smell_commits WHERE session_id=15 AND status='running';")
  failed=$(sqlite3 "$DB" "SELECT count(*) FROM smell_commits WHERE session_id=15 AND status='failed';")
  delta=$((llm - prev))
  ts=$(date '+%H:%M:%S')
  echo "[$ts] phase2: $done/$total | llm_results: $llm (+$delta since last) | running: $running | failed: $failed"
  if [ "$delta" -eq 0 ] && [ "$llm" -gt 0 ]; then
    echo "  ⚠️  Nessun nuovo risultato negli ultimi 30s — potrebbe essere bloccato"
  fi
  prev=$llm
  sleep 30
done
