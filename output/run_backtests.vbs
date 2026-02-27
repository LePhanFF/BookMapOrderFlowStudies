Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

strBase = "C:\Users\lehph\Documents\GitHub\BookMapOrderFlowStudies-2"
strPython = "E:\anaconda\python.exe"
strScript = strBase & "\scripts\run_backtest.py"
strOut = strBase & "\output\"

' Command 1: OR Reversal
strCmd1 = """" & strPython & """ """ & strScript & """ --strategies ""Opening Range Rev"" --instrument NQ"
Set objExec1 = objShell.Exec(strCmd1)
Do While objExec1.Status = 0
    WScript.Sleep 100
Loop
strOutput1 = objExec1.StdOut.ReadAll()
strErr1 = objExec1.StdErr.ReadAll()

Set objFile1 = objFSO.CreateTextFile(strOut & "cmd1_result.txt", True)
objFile1.Write strOutput1
If Len(strErr1) > 0 Then
    objFile1.Write Chr(10) & "--- STDERR ---" & Chr(10) & strErr1
End If
objFile1.Close

' Command 2: OR Acceptance
strCmd2 = """" & strPython & """ """ & strScript & """ --strategies ""OR Acceptance"" --instrument NQ"
Set objExec2 = objShell.Exec(strCmd2)
Do While objExec2.Status = 0
    WScript.Sleep 100
Loop
strOutput2 = objExec2.StdOut.ReadAll()
strErr2 = objExec2.StdErr.ReadAll()

Set objFile2 = objFSO.CreateTextFile(strOut & "cmd2_result.txt", True)
objFile2.Write strOutput2
If Len(strErr2) > 0 Then
    objFile2.Write Chr(10) & "--- STDERR ---" & Chr(10) & strErr2
End If
objFile2.Close

' Fallback if command 2 failed
If objExec2.ExitCode <> 0 Then
    strCmd2b = """" & strPython & """ """ & strScript & """ --strategies ""Acceptance"" --instrument NQ"
    Set objExec2b = objShell.Exec(strCmd2b)
    Do While objExec2b.Status = 0
        WScript.Sleep 100
    Loop
    strOutput2b = objExec2b.StdOut.ReadAll()
    strErr2b = objExec2b.StdErr.ReadAll()
    Set objFile2b = objFSO.CreateTextFile(strOut & "cmd2_fallback_result.txt", True)
    objFile2b.Write strOutput2b
    If Len(strErr2b) > 0 Then
        objFile2b.Write Chr(10) & "--- STDERR ---" & Chr(10) & strErr2b
    End If
    objFile2b.Close
End If

Set objDoneFile = objFSO.CreateTextFile(strOut & "run_done.txt", True)
objDoneFile.Write "DONE"
objDoneFile.Close

WScript.Echo "Completed successfully"
