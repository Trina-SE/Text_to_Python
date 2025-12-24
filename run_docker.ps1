param(
    [string]$Device = "cpu",
    [int]$Epochs = 10,
    [int]$TrainSize = 8000,
    [int]$ValSize = 1000,
    [int]$TestSize = 1000
)

docker build -t text2python .

docker run --rm `
    -v "$PWD\outputs:/app/outputs" `
    -v "$PWD\checkpoints:/app/checkpoints" `
    -v "$PWD\figures:/app/figures" `
    text2python `
    --device $Device `
    --epochs $Epochs `
    --train-size $TrainSize `
    --val-size $ValSize `
    --test-size $TestSize
