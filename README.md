# The criterium_ignore addon for torch/nn

The package is for use with [torch/nn](https://github.com/torch/nn) and adds a
method for ignoring labels. It is a direct extension of the [ParallelCriterion][1]
where the `:add()` allows you to  specify an ignore label for each criterion that you add.

As of version 0.2 you now also have the power of `argcheck` for help with arguments
etc. If you mistype an argument then there is an automated help print.

[1]: https://github.com/torch/nn/blob/master/doc/criterion.md#nn.ParallelCriterion

## Installation

In order to install the package you need to do it directly from the GitHub repo (at the moment):

```bash
luarocks install https://raw.githubusercontent.com/gforge/criterion_ignore/master/criterion_ignore-0.2-1.rockspec
```

## Use case:

```lua
require 'criterion_ignore'
model = nn.Sequential()
model:add(nn.Linear(3,5))

criterion = criterion_ignore.Parallel.new()
prl = nn.ConcatTable()
for i=1,7 do
    seq = nn.Sequential()
    seq:add(nn.Linear(5,i + 1))
    seq:add(nn.SoftMax())
    prl:add(seq)
    -- First parameter is weight while the second is the ignore label
    criterion:add{
        criterion = nn.ClassNLLCriterion(),
        ignore = 0
    }
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
