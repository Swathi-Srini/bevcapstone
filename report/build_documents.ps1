# Build both documents (twice each, so that the table of contents and cross references resolve).
# Uses a clean PATH because the Git-Bash PATH on this machine contains an invalid directory entry that makes pdflatex fail.
# Run from anywhere:   powershell -File D:\bev_Fresh\bevcapstone\report\build_documents.ps1
$miktex = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
$env:Path = "$miktex;C:\Windows\System32;C:\Windows"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
foreach ($doc in @("Directors_Research_Guide", "Complete_Technical_Research_Report", "Five_Model_Comparison")) {
    for ($pass = 1; $pass -le 2; $pass++) {
        & "$miktex\pdflatex.exe" -interaction=nonstopmode "$doc.tex" > "build_$doc.log" 2>&1
        "$doc pass $pass exit code $LASTEXITCODE"
    }
    $errors = Select-String -Path "build_$doc.log" -Pattern "^!" | Select-Object -First 5
    $undefined = Select-String -Path "build_$doc.log" -Pattern "Reference .* undefined|There were undefined references" | Select-Object -First 5
    $pages = Select-String -Path "build_$doc.log" -Pattern "Output written" | ForEach-Object { $_.Line }
    "$doc : $pages"
    if ($errors) { "ERRORS:"; $errors | ForEach-Object { $_.Line } }
    if ($undefined) { "UNDEFINED REFERENCES:"; $undefined | ForEach-Object { $_.Line } }
}
