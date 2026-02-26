$ErrorActionPreference = "Stop"

$runDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $runDir

$repoRoot = Resolve-Path (Join-Path $runDir "..\..")
$grapheneFc = Join-Path $repoRoot "studies\graphene_bulk_2026q1\inputs\ifc\graphene.fc"
$scipyDll = "C:\Users\zhbho\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\scipy.libs\libscipy_openblas-f07f5a5d207a3a47104dca54d6d0c86a.dll"

if (-not (Test-Path $grapheneFc)) {
  throw "Missing graphene.fc at $grapheneFc"
}

if (-not (Test-Path ".\a.out.exe")) {
  if (Test-Path ".\a.out") {
    Copy-Item ".\a.out" ".\a.out.exe" -Force
  } else {
    throw "Missing a.out in run directory."
  }
}

New-Item -ItemType Directory -Force -Path ".\force" | Out-Null
Copy-Item $grapheneFc ".\graphene_1L_PBE_van.fc" -Force

if (-not (Test-Path ".\get_K.exe")) {
  gfortran -o get_K get_K.f90
}

.\get_K
Move-Item -Force ".\Kmatrix.dat" ".\force\Kmatrix.dat"

Copy-Item $scipyDll ".\libscipy_openblas.dll" -Force
Copy-Item "C:\Strawberry\c\bin\libgfortran-5.dll" ".\libgfortran-5.dll" -Force
Copy-Item "C:\Strawberry\c\bin\libquadmath-0.dll" ".\libquadmath-0.dll" -Force
Copy-Item "C:\Strawberry\c\bin\libgcc_s_seh-1.dll" ".\libgcc_s_seh-1.dll" -Force

.\a.out.exe
