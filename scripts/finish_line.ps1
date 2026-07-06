# finish_line.ps1 - runs the entire post-rebuild finish line in one go.
#
#   powershell -ExecutionPolicy Bypass -File scripts\finish_line.ps1
#
# Run from the repo root with the venv activated, AFTER ingest_corpus.py
# --rebuild has printed its final "Done:" summary. Safe to re-run: every
# step is idempotent (registration skips known docs, ingest skips
# up-to-date docs, evals just overwrite their reports).

param(
    [string]$Label = "v2.0",
    [string]$CorpusDir = "my_pdfs",
    [int]$Port = 8123
)
$ErrorActionPreference = "Stop"

Write-Host "== Chempair KB finish line ($Label) =="
$answer = Read-Host "Has the rebuild terminal printed its final 'Done:' summary? (y/n)"
if ($answer -ne "y") {
    Write-Host "Wait for the rebuild to finish, then re-run this script."
    exit 1
}

Write-Host "`n[1/7] Pulling latest repo changes..."
git pull
if ($LASTEXITCODE -ne 0) { Write-Host "git pull failed - resolve and re-run."; exit 1 }

Write-Host "`n[2/7] Registering any new PDFs..."
python scripts/corpus_manifest.py seed --corpus-dir $CorpusDir
python scripts/corpus_manifest.py validate --corpus-dir $CorpusDir
if ($LASTEXITCODE -ne 0) { Write-Host "Manifest validation failed - fix the errors above and re-run."; exit 1 }

Write-Host "`n[3/7] Ingesting new documents (skips everything up to date)..."
python ingest_corpus.py
if ($LASTEXITCODE -ne 0) { Write-Host "Ingest reported failures - check reports\ingest\ and re-run this script."; exit 1 }

Write-Host "`n[4/7] Embedding bake-off (first run downloads two candidate models)..."
python evals/retrieval_eval.py --rag-storage ./rag_storage
if ($LASTEXITCODE -ne 0) { Write-Host "Bake-off failed - see output above."; exit 1 }

Write-Host "`n[5/7] Post-rebuild answer eval against a temporary local server..."
$env:RAG_STORAGE = "./rag_storage"
$env:RAG_AUTH_REQUIRED = "false"
$env:PORT = "$Port"
$server = Start-Process -FilePath "python" -ArgumentList "start.py" -PassThru -WindowStyle Hidden
try {
    $healthy = $false
    foreach ($attempt in 1..60) {
        Start-Sleep -Seconds 2
        try {
            $health = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3
            if ($health.status -eq "ok") { $healthy = $true; break }
        } catch { }
    }
    if (-not $healthy) { throw "Local server did not become healthy on port $Port" }
    python evals/run_eval.py --base-url "http://127.0.0.1:$Port" --out evals/baseline/full-post-rebuild/
} finally {
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
}

Write-Host "`n[6/7] Packaging KB $Label..."
python scripts/package_kb.py --rag-storage ./rag_storage --label $Label
if ($LASTEXITCODE -ne 0) { Write-Host "Packaging failed - see output above."; exit 1 }

Write-Host "`n[7/7] Committing the evidence..."
git add corpus/manifest.yaml evals/
git commit -m "KB ${Label}: corpus registered + rebuilt; post-rebuild eval and embedding bake-off results"
if ($LASTEXITCODE -ne 0) { Write-Host "(nothing new to commit - fine)" }
git push

Write-Host @"

=====================================================================
 ALL DONE on this machine. Two manual clicks remain:

 1. github.com/JNSteve/chempair-RAG -> Releases -> Draft a new release
    tag: $Label  -> attach dist\rag_storage.tar.gz -> Publish
 2. Railway -> web service -> Variables:
       KB_RELEASE_URL = <the new asset's download URL>
       RAG_LLM_MODEL  = gpt-5.4
    then clear the data volume (delete vdb_entities.json) and redeploy.

 Verify: /health shows "label": "$Label" and the new model.
 Then tell Claude the script finished - the reports are already pushed.
=====================================================================
"@
