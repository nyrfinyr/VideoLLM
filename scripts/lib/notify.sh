#!/bin/bash
# Notifiche Telegram sul ciclo di vita dei job SLURM.
#
# UN messaggio per job, MODIFICATO in place lungo tutto il ciclo:
#   🕓 PENDING   creato da `scripts/tgsbatch` al momento della sottomissione
#   🏃 RUNNING   creato/aggiornato qui, all'avvio dello script sbatch
#   ✅ ❌ ⏱      aggiornato qui dal trap di uscita, con le metriche finali
#
# Il passaggio PENDING → RUNNING non richiede nessun demone né polling di
# `squeue`: lo script sbatch INIZIA a girare esattamente quando SLURM manda il
# job in esecuzione, quindi `tg_job_start` È la transizione.
#
# Uso, una riga per sbatch subito dopo il `cd` nel repo:
#   source scripts/lib/notify.sh && tg_job_start
#
# Configurazione in ~/.config/telegram-notify.env (chmod 600, MAI nel repo):
#   TG_BOT_TOKEN=123456:AAH...
#   TG_CHAT_ID=987654321
# Se il file manca o le due variabili non ci sono, TUTTO diventa no-op: i job
# girano identici a prima per chiunque non abbia configurato il bot.
#
# VINCOLO: gli sbatch girano con `set -euo pipefail`. Nessuna funzione qui
# dentro può ritornare non-zero o far fallire il job — ogni `curl` ha
# `--max-time` ed è chiuso da `|| true`, ogni funzione termina con `return 0`.
# Le metriche NON sono elencate qui: le scrive `utils/notify.py` leggendo
# `wandb.run.summary` per intero (vedi il file per il perché).

# Sourcing multiplo: non reinstallare i trap né rigenerare il file di summary.
[[ -n "${_TG_NOTIFY_SOURCED:-}" ]] && return 0
_TG_NOTIFY_SOURCED=1

TG_ENV_FILE="${TELEGRAM_NOTIFY_ENV:-$HOME/.config/telegram-notify.env}"
TG_STATE_DIR="${TG_STATE_DIR:-$HOME/.cache/telegram-notify}"
_TG_API="${TG_API_BASE:-https://api.telegram.org}"  # override: mock locale nei test
_TG_CURL=(curl -sS --max-time 10 --retry 2 --retry-delay 1)

# --- configurazione -------------------------------------------------------

_tg_load_env() {
    if [[ -r "$TG_ENV_FILE" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "$TG_ENV_FILE" 2>/dev/null || true
        set +a
    fi
    [[ -n "${TG_BOT_TOKEN:-}" && -n "${TG_CHAT_ID:-}" ]]
}

if _tg_load_env; then _TG_ENABLED=1; else _TG_ENABLED=0; fi

# Predicato pubblico: `scripts/tgsbatch` lo usa per decidere se degradare a
# `sbatch` puro.
tg_enabled() { [[ "$_TG_ENABLED" == "1" ]]; }

# --- primitive Telegram ---------------------------------------------------

# Escape dei soli caratteri che rompono parse_mode=HTML.
tg_escape() { sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'; }

# Manda un messaggio; stampa su stdout il message_id (vuoto se fallisce).
# `grep -o` invece di `jq`: jq non è garantito su tutti i nodi. Lo spazio
# opzionale attorno ai due punti è tollerato apposta — Telegram risponde
# compatto, ma un proxy che riformatta il JSON non deve far perdere l'id (e
# senza id ogni aggiornamento diventerebbe un messaggio nuovo).
tg_send() {
    tg_enabled || return 0
    local resp
    resp=$("${_TG_CURL[@]}" -X POST "$_TG_API/bot${TG_BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${TG_CHAT_ID}" \
        --data-urlencode "text=$1" \
        -d "parse_mode=HTML" -d "disable_web_page_preview=true" 2>/dev/null) || true
    printf '%s' "$resp" \
        | grep -o '"message_id"[[:space:]]*:[[:space:]]*[0-9]\+' | head -1 | tr -cd '0-9'
    return 0
}

# Modifica un messaggio esistente. Se l'edit fallisce (messaggio cancellato a
# mano, contenuto identico, message_id perso) ripiega su un messaggio nuovo:
# meglio un messaggio in più che una notifica persa.
tg_edit() {
    local mid="$1" text="$2" resp
    tg_enabled || return 0
    if [[ -z "$mid" ]]; then tg_send "$text" >/dev/null; return 0; fi
    resp=$("${_TG_CURL[@]}" -X POST "$_TG_API/bot${TG_BOT_TOKEN}/editMessageText" \
        --data-urlencode "chat_id=${TG_CHAT_ID}" \
        -d "message_id=${mid}" \
        --data-urlencode "text=${text}" \
        -d "parse_mode=HTML" -d "disable_web_page_preview=true" 2>/dev/null) || true
    if ! printf '%s' "$resp" | grep -q '"ok"[[:space:]]*:[[:space:]]*true'; then
        # "message is not modified" non è un errore da rimediare con un doppione.
        printf '%s' "$resp" | grep -q 'message is not modified' || tg_send "$text" >/dev/null
    fi
    return 0
}

# --- stato su disco -------------------------------------------------------
#
# Una directory per SOTTOMISSIONE (array job id, o job id per i job singoli),
# dentro un file per task. Serve a due cose: ritrovare il message_id fra
# `tgsbatch` (login node) e lo script sbatch (nodo di calcolo), e contare i
# task finiti per il riepilogo dell'array.

tg_state_dir() {
    printf '%s/%s' "$TG_STATE_DIR" "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local-$$}}"
}

tg_task_key() { printf '%s' "${SLURM_ARRAY_TASK_ID:-main}"; }

# --- formattazione --------------------------------------------------------

_tg_dur() {
    local s=$1
    if   (( s < 60 ));   then printf '%ds' "$s"
    elif (( s < 3600 )); then printf '%dm' $(( s / 60 ))
    else printf '%dh%02dm' $(( s / 3600 )) $(( (s % 3600) / 60 ))
    fi
}

# Identità leggibile del job: `nome · 74423_1` per gli array, `nome · 74423`
# per i job singoli.
_tg_job_label() {
    local id="${SLURM_JOB_ID:-?}"
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    printf '%s · %s' "${SLURM_JOB_NAME:-job}" "$id"
}

# Path dello stderr del job, per allegarne la coda quando fallisce: gli sbatch
# lo dirigono su `logs/%x-%j.err` ma il path espanso lo conosce solo SLURM.
_tg_stderr_path() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || return 0
    scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
        | tr ' ' '\n' | sed -n 's/^StdErr=//p' | head -1
    return 0
}

# --- ciclo di vita --------------------------------------------------------

# Da chiamare come primo comando dello sbatch: notifica RUNNING e arma i trap.
tg_job_start() {
    tg_enabled || return 0
    local dir gpu queued="" now text
    _TG_T0=$(date +%s)
    dir=$(tg_state_dir)
    mkdir -p "$dir" 2>/dev/null || true
    _TG_MSGID_FILE="$dir/$(tg_task_key).msgid"

    # File che `utils/notify.py` riempirà con TUTTE le metriche di wandb.
    # Esportato: lo legge il processo Python figlio.
    if [[ -z "${TG_SUMMARY_FILE:-}" ]]; then
        TG_SUMMARY_FILE=$(mktemp -t tg-summary.XXXXXX 2>/dev/null || printf '%s/summary.%s' "$dir" "$(tg_task_key)")
        export TG_SUMMARY_FILE
    fi
    : > "$TG_SUMMARY_FILE" 2>/dev/null || true

    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) || true
    if [[ -r "$dir/submit.ts" ]]; then
        now=$(date +%s)
        queued=" · in coda $(_tg_dur $(( now - $(cat "$dir/submit.ts") )))"
    fi

    text="🏃 <b>RUNNING</b> · $(printf '%s' "$(_tg_job_label)" | tg_escape)
$(printf '%s' "node ${SLURMD_NODENAME:-$(hostname)}${gpu:+ · $gpu}${queued}" | tg_escape)"

    # L'array condivide un solo messaggio PENDING (uno per sottomissione), ma
    # ogni TASK ha il suo messaggio: job nuovo → messaggio nuovo. Il PENDING
    # viene riusato solo dal job singolo, che è lo stesso job.
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" && -r "$dir/submit.msgid" ]]; then
        cp "$dir/submit.msgid" "$_TG_MSGID_FILE" 2>/dev/null || true
    fi
    if [[ -r "$_TG_MSGID_FILE" ]]; then
        tg_edit "$(cat "$_TG_MSGID_FILE")" "$text"
    else
        tg_send "$text" > "$_TG_MSGID_FILE" 2>/dev/null || true
    fi

    trap '_tg_finish "$?"' EXIT
    trap '_tg_finish 143 TERM' TERM
    trap '_tg_finish 130 INT' INT
    return 0
}

# Trap di uscita: aggiorna IL MEDESIMO messaggio con l'esito e le metriche.
# Copre anche timeout SLURM e `scancel` (SIGTERM al batch shell prima del
# SIGKILL) e l'OOM killer (uccide Python, non la shell → ramo EXIT con rc≠0).
_tg_finish() {
    local rc="${1:-0}" sig="${2:-}" head body errf tail_ dur mid=""
    [[ -n "${_TG_DONE:-}" ]] && return 0
    _TG_DONE=1
    tg_enabled || return 0
    dur=$(_tg_dur $(( $(date +%s) - ${_TG_T0:-$(date +%s)} )))

    case "$sig:$rc" in
        TERM:*) head="⏱ <b>TIMEOUT/CANCEL</b>" ;;
        INT:*)  head="🛑 <b>INTERROTTO</b>" ;;
        *:0)    head="✅ <b>DONE</b>" ;;
        *)      head="❌ <b>FAIL</b> rc=${rc}" ;;
    esac
    body="${head} · $(printf '%s' "$(_tg_job_label)" | tg_escape) · ${dur}
$(printf '%s' "node ${SLURMD_NODENAME:-$(hostname)}" | tg_escape)"

    # Metriche: qualunque cosa utils/notify.py abbia scritto, senza sapere cosa.
    if [[ -s "${TG_SUMMARY_FILE:-/nonexistent}" ]]; then
        body="${body}
<pre>$(tg_escape < "$TG_SUMMARY_FILE")</pre>"
    fi

    # Sui fallimenti la coda dello stderr è quasi sempre la risposta.
    if [[ "$rc" != "0" ]]; then
        errf=$(_tg_stderr_path)
        if [[ -n "$errf" && -r "$errf" ]]; then
            tail_=$(tail -c 1200 "$errf" 2>/dev/null | tail -n 15) || true
            [[ -n "$tail_" ]] && body="${body}
<pre>$(printf '%s' "$tail_" | tg_escape)</pre>"
        fi
    fi

    [[ -r "${_TG_MSGID_FILE:-/nonexistent}" ]] && mid=$(cat "$_TG_MSGID_FILE")
    tg_edit "$mid" "$body"
    _tg_record_result "$rc" "$sig"
    return 0
}

# Riepilogo dell'array: registra l'esito del task e riscrive il conteggio nel
# messaggio di sottomissione. Ricontare da zero ogni volta è idempotente —
# anche se due task si accavallano nonostante il flock, l'ultimo che finisce
# scrive il numero giusto.
#
# SOLO per gli array: un job singolo ha già RIUSATO il messaggio di
# sottomissione come proprio messaggio di stato (vedi `tg_job_start`), quindi
# riscriverlo qui cancellerebbe le metriche appena pubblicate.
_tg_record_result() {
    local rc="$1" sig="$2" dir mid total ok fail state
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || return 0
    dir=$(tg_state_dir)
    [[ -d "$dir" && -r "$dir/submit.msgid" && -r "$dir/submit.text" ]] || return 0
    case "$sig:$rc" in TERM:*|INT:*) state=timeout ;; *:0) state=ok ;; *) state=fail ;; esac
    printf '%s\n' "$state" > "$dir/$(tg_task_key).result" 2>/dev/null || true

    (
        flock -w 10 9 2>/dev/null || true
        ok=$(grep -lx ok "$dir"/*.result 2>/dev/null | wc -l)
        fail=$(grep -Lx ok "$dir"/*.result 2>/dev/null | wc -l)
        total=$(cat "$dir/total" 2>/dev/null || printf '?')
        mid=$(cat "$dir/submit.msgid")
        tg_edit "$mid" "$(cat "$dir/submit.text")
$(printf '%s' "conclusi $(( ok + fail ))/${total} · ✅ ${ok} · ❌ ${fail}" | tg_escape)"
    ) 9>"$dir/.lock" 2>/dev/null || true
    return 0
}

return 0
