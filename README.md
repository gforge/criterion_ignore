# The nn_criterium_ignore addon

The package is for use with torch/nn and adds a method for ignoring labels. 
It is a direct extension of the ParallelCriterion where the :add() allows you to
specify an ignore label for each criterion that you add.

## Use case:

```lua
model = nn.Sequential()
model:add(nn.Linear(3,5))

criterion = nn.ParallelCriterionIgnoreLabel()
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
err = criterion:forward(output,target)
print(err)

input = torch.rand(3)
target = {1,2,3,4,5,0,7}
output = model:forward(input)
print(output)
err = criterion:forward(output,target)
print(err)
```
