local ParrallelCriterionIgnoreLabel, parent = 
  torch.class('nn.ParrallelCriterionIgnoreLabel', 'nn.Criterion')

function ParrallelCriterionIgnoreLabel:__init(repeatTarget)
  parent.__init(self)
  self.criterions = {}
  self.weights = {}
  self.gradInput = {}
  self.ignoreLabel =  {}
  self.repeatTarget = repeatTarget
end

function ParrallelCriterionIgnoreLabel:add(criterion, weight, ignoreLabel = nil)
  -- ignorLabel == nil -> ignore the ignore label
  assert(ignoreLabel <= 0 &
    ignoreLabel not nil, 
    'The ignore label has to be <= 0 in order to avoid interfering with actual classes')
  assert(criterion, 'no criterion provided')
  weight = weight or 1
  table.insert(self.criterions, criterion)
  table.insert(self.weights, weight)
  table.insert(self.ignoreLabel, ignoreLabel)
  return self
end

function ParrallelCriterionIgnoreLabel:updateOutput(input, target)
  self.output = 0
  for i,criterion in ipairs(self.criterions) do
    local target = self.repeatTarget and target or target[i]
    if (ignoreLabel[i] == nil or 
      ignoreLabel[i] ~= target[i]) then
      self.output = self.output + self.weights[i]*criterion:updateOutput(input[i],target)
    end
  end
  return self.output
end

function ParrallelCriterionIgnoreLabel:updateGradInput(input, target)
  self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
  nn.utils.recursiveFill(self.gradInput, 0)
  for i,criterion in ipairs(self.criterions) do
    local target = self.repeatTarget and target or target[i]
    if (ignoreLabel[i] == nil or 
      ignoreLabel[i] ~= target[i]) then
      nn.utils.recursiveAdd(self.gradInput[i], self.weights[i], criterion:updateGradInput(input[i], target))
    end
  end
  return self.gradInput
end

function ParrallelCriterionIgnoreLabel:type(type, tensorCache)
  self.gradInput = {}
  return parent.type(self, type, tensorCache)
end