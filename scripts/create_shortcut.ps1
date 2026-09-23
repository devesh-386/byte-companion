# Creates a "Byte" shortcut on the Desktop that starts the companion without a console window.
$root = Split-Path -Parent $PSScriptRoot
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "Byte.lnk"))
$lnk.TargetPath = Join-Path $root ".venv\Scripts\pythonw.exe"
$lnk.Arguments = "-m companion.ui"
$lnk.WorkingDirectory = $root
$lnk.IconLocation = (Join-Path $root "assets\byte.ico") + ",0"
$lnk.Description = "Byte - local AI companion"
$lnk.Save()
Write-Output "Shortcut created: $($lnk.FullName)"
