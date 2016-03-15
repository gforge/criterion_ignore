# The criterium_ignore addon for torch/nn

The package is for use with [torch/nn](https://github.com/torch/nn) and adds a method for ignoring labels. 
It is a direct extension of the ParallelCriterion where the `:add()` allows you to
specify an ignore label for each criterion that you add.

## Use case:

```lua
require 'criterion_ignore'
model = nn.Sequential()
model:add(nn.Linear(3,5))

criterion = criterion_ignore.Parallel()
prl = nn.ConcatTable()
for i=1,7 do
    seq = nn.Sequential()
    seq:add(nn.Linear(5,i + 1))
    seq:add(nn.SoftMax())
    prl:add(seq)
    -- First parameter is weight while the second is the ignore label
    criterion:add(nn.ClassNLLCriterion(), 1, 0)
end
model:add(prl)

input = torch.rand(3)
target = {1,2,3,4,5,6,7}
output = model:forward(input)
print(output)
err1 = criterion:forward(output,target)
print(err1)

target[5] = 0
output = model:forward(input)
print(output)
err2 = criterion:forward(output,target)
print(err2)
print(err1 < err2)
```
