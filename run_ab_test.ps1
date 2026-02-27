$outFile = "C:\Users\lehph\Documents\GitHub\BookMapOrderFlowStudies-2\output\ab_run_output.txt"
& "E:\anaconda\python.exe" "C:\Users\lehph\Documents\GitHub\BookMapOrderFlowStudies-2\scripts\test_or_rev_ab.py" 2>&1 | Out-File -FilePath $outFile -Encoding utf8
Write-Host "Done. Exit: $LASTEXITCODE"
