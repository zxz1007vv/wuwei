
# Setup Python environment for Wuwei AI, Change If Needed
conda activate nn
& "$PSScriptRoot/../../alpha_zero/._env/Scripts/activate.ps1"

Set-Location $PSScriptRoot/..
# Start Wuwei
if ($args.Count -eq 0) {
    Write-Debug "Start Wuwei AI with Policy Network Only"
    python main.py gtp
} elseif ( $args[1] -eq "MCTS" ) {
    Write-Debug "Start Wuwei AI with MCTS"
    python main.py gtp MCTS
} else {
    Write-Error "Invalid argument. Use 'MCTS' or No argument."
}