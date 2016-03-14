require 'torch'
require 'nn'
local ParallelCriterionIgnoreLabel, parent = torch.class('nn.ParallelCriterionIgnoreLabel', 'nn.Criterion')

function ParallelCriterionIgnoreLabel:__init(repeatTarget)
  parent.__init(self)
  self.criterions = {}
  self.weights = {}
  self.gradInput = {}
  self.ignoreLabel = {}
  self.repeatTarget = repeatTarget
end

function ParallelCriterionIgnoreLabel:add(criterion, weight, ignore)
  -- ignore == nil -> ignore the ignore label
  ignore = ignore or "none"
  if (not ignore == "none") then
    assert(ignore <= 0, 
      'The ignore label has to be <= 0 in order to avoid interfering with actual classes.' ..
      ' Current value is: ' .. ignore)
  end
  assert(criterion, 'no criterion provided')
  weight = weight or 1
  table.insert(self.criterions, criterion)
  table.insert(self.weights, weight)
  table.insert(self.ignoreLabel, ignore)
  return self
end

function ParallelCriterionIgnoreLabel:updateOutput(input, target)
  self.output = 0
  for i,criterion in ipairs(self.criterions) do
    local target = self.repeatTarget and target or target[i]
    if (ignoreLabel[i] == "none" or 
      ignoreLabel[i] ~= target[i]) then
      self.output = self.output + self.weights[i]*criterion:updateOutput(input[i],target)
    end
  end
  return self.output
end

function ParallelCriterionIgnoreLabel:updateGradInput(input, target)
  self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
  nn.utils.recursiveFill(self.gradInput, 0)
  for i,criterion in ipairs(self.criterions) do
    local target = self.repeatTarget and target or target[i]
    if (ignoreLabel[i] == "none" or 
      ignoreLabel[i] ~= target[i]) then
      nn.utils.recursiveAdd(self.gradInput[i], self.weights[i], criterion:updateGradInput(input[i], target))
    end
  end
  return self.gradInput
end

function ParallelCriterionIgnoreLabel:type(type, tensorCache)
  self.gradInput = {}
  return parent.type(self, type, tensorCache)
end

return ParallelCriterionIgnoreLabel