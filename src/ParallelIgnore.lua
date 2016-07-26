require('torch')
local nn = require('nn')
local argcheck = require('argcheck')

local ParallelIgnore, parent =
  torch.class('nn.ParallelIgnore', 'nn.Criterion')

ParallelIgnore.__init = argcheck{
  doc = [[
### nn.ParallelIgnore(@ARGP)

The init function

@ARGT
]],
  {name='self', type='nn.Criterion'},
  {name='repeatTarget', type='boolean', default=false,
   doc=[[If repeatTarget=true, the target is repeatedly presented to
   each criterion (with a different input)]]},
  call=function(self, repeatTarget)
  parent.__init(self)
  self.criterions = {}
  self.weights = {}
  self.gradInput = {}
  self.ignoreLabel = {}
  self.repeatTarget = repeatTarget
end}

ParallelIgnore.add = argcheck{
  doc = [[
### nn.ParallelIgnore(@ARGP)

Add a criterion

@ARGT

_Return value_: self
]],
  {name='self', type='nn.Criterion'},
  {name='criterion', type='nn.Criterion', doc='The criterion to add'},
  {name='weight', type='number', default=1,
   doc='The criterion weight that the output gets multiplied by'},
  {name='ignore', type='number', default='none',
   doc='The target value (must be <= 0) that should be ignored when summing.'},
  call=function(self, criterion, weight, ignore)
  if (not ignore == "none") then
    assert(ignore <= 0,
      'The ignore label has to be <= 0 in order to avoid interfering with actual classes.' ..
      ' Current value is: ' .. ignore)
  end

  table.insert(self.criterions, criterion)
  table.insert(self.weights, weight)
  table.insert(self.ignoreLabel, ignore)
  return self
end}

ParallelIgnore.updateOutput = argcheck{
  doc = [[
### nn.updateOutput(@ARGP)

Upades the self.output a weighted sum of the criterions `updateOutput`

@ARGT

_Return value_: self.output
]],
  {name='self', type='nn.Criterion'},
  {name='input', type='table|torch.*Tensor', opt=true,
   doc='The input data (the network guess)'},
  {name='target', type='table|torch.*Tensor', opt=true,
   doc='The target data (the true label)'},
  call=function(self, input, target)
  self.output = 0
  for i,criterion in ipairs(self.criterions) do
    local target = self.repeatTarget and target or target[i]
    if (self.ignoreLabel[i] == "none" or
      self.ignoreLabel[i] ~= target) then
      self.output = self.output + self.weights[i]*criterion:updateOutput(input[i],target)
    end
  end
  return self.output
end}

ParallelIgnore.updateGradInput = argcheck{
  doc = [[
### nn.updateGradInput(@ARGP)

Upades the self.gradInput a weighted sum of the criterions `updateGradInput`

@ARGT

_Return value_: self.gradInput
]],
  {name='self', type='nn.Criterion'},
  {name='input', type='table|torch.*Tensor', doc='The input data'},
  {name='target', type='table|torch.*Tensor', doc='The target data'},
  call=function(self, input, target)
  self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
  nn.utils.recursiveFill(self.gradInput, 0)
  for i,criterion in ipairs(self.criterions) do
    local target = self.repeatTarget and target or target[i]
    if (self.ignoreLabel[i] == "none" or
      self.ignoreLabel[i] ~= target) then
      nn.utils.recursiveAdd(self.gradInput[i], self.weights[i], criterion:updateGradInput(input[i], target))
    end
  end
  return self.gradInput
end}

ParallelIgnore.type = argcheck{
  doc = [[
### nn.type(@ARGP)

Sets the type

@ARGT

_Return value_: self
]],
  {name='self', type='nn.Criterion'},
  {name='type', type='string', doc='The torch tensor type'},
  {name='tensorCache', type='table|torch.*Tensor', opt=true,
   doc=[[tensorCache maintains a list of all tensors and storages that have been
   converted (recursively) by calls to recursiveType() and type()]]},
  call=function(self, type, tensorCache)
  self.gradInput = {}
  return parent.type(self, type, tensorCache)
end}

return ParallelIgnore
