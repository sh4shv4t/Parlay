# Find-Python311.ps1 — dot-source from setup scripts
# Returns $null or a hashtable: Exe, PreArgs, Ver, Name

function Get-Python311 {
    $candidateDefs = [System.Collections.Generic.List[object]]::new()
    $null = $candidateDefs.Add(@{ Name = "py -3.11 (Windows launcher)"; Exe = "py"; PreArgs = @("-3.11") })
    $null = $candidateDefs.Add(@{ Name = "py -3.11-64 (launcher tag)"; Exe = "py"; PreArgs = @("-3.11-64") })
    $null = $candidateDefs.Add(@{ Name = "python3.11"; Exe = "python3.11"; PreArgs = @() })
    $null = $candidateDefs.Add(@{ Name = "python3"; Exe = "python3"; PreArgs = @() })
    $null = $candidateDefs.Add(@{ Name = "python"; Exe = "python"; PreArgs = @() })
    $local311 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
    $null = $candidateDefs.Add(@{ Name = "per-user: $local311"; Exe = $local311; PreArgs = @() })
    $allUsers = "${env:ProgramFiles}\Python311\python.exe"
    $null = $candidateDefs.Add(@{ Name = "all-users: $allUsers"; Exe = $allUsers; PreArgs = @() })

    function Test-One {
        param([string] $Exe, [string[]] $PreArgs)
        if ($Exe -match '^(?:[A-Za-z]:|\\\\)') {
            if (-not (Test-Path -LiteralPath $Exe)) { return $null }
        } else {
            if (-not (Get-Command $Exe -ErrorAction SilentlyContinue)) { return $null }
        }
        try {
            $verOut = & $Exe @($PreArgs + @("--version")) 2>&1 | Out-String
        } catch {
            return $null
        }
        if ($verOut -match "3\.11\.\d+") {
            return @{ Exe = $Exe; PreArgs = $PreArgs; Ver = $verOut.Trim() }
        }
        return $null
    }

    foreach ($c in $candidateDefs) {
        $r = Test-One -Exe $c.Exe -PreArgs $c.PreArgs
        if ($r) {
            $r["Name"] = $c.Name
            return $r
        }
    }
    return $null
}

function Get-Python311Diagnostics {
    $lines = [System.Collections.Generic.List[string]]::new()
    $candidateDefs = @(
        @{ Name = "py -3.11"; Exe = "py"; PreArgs = @("-3.11") }
        @{ Name = "python"; Exe = "python"; PreArgs = @() }
        @{ Name = "python3"; Exe = "python3"; PreArgs = @() }
    )
    $local311 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
    $candidateDefs += @{ Name = "per-user"; Exe = $local311; PreArgs = @() }

    foreach ($c in $candidateDefs) {
        if ($c.Exe -match '^(?:[A-Za-z]:|\\\\)') {
            if (-not (Test-Path -LiteralPath $c.Exe)) { continue }
        } elseif (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
        try {
            $v = & $c.Exe @($c.PreArgs + @("--version")) 2>&1 | Out-String
            $null = $lines.Add("$($c.Name) => $($v.Trim())")
        } catch { }
    }
    return $lines
}
