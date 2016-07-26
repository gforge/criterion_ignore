-- you can easily test specific units like this:
-- th -lnn -e "nn.test{'ParallelCriterionIgnoreLabel'}"
require 'nn'

-- Make sure that directory structure is always the same
local init_file = paths.thisfile():gsub("/test/.*.lua", "/init.lua")

-- Include Dataframe lib
local criterion_ignore = paths.dofile(init_file)

local mytester = torch.Tester()
local jac

local precision = 1e-5
local expprecision = 1e-4

local criterion_ignore_test = torch.TestSuite()

--[[ Generate tests to exercise the tostring component of modules. ]]
local tostringTestModules = {
  nnLinear = nn.Linear(1, 2),
  nnReshape = nn.Reshape(10),
  nnSpatialZeroPadding = nn.SpatialZeroPadding(1, 1, 1, 1)}
for test_name, component in pairs(tostringTestModules) do
  criterion_ignore_test['tostring' .. test_name] =
    function ()
      mytester:assert(tostring(component):find(
                         torch.type(component) .. '(', 1, true) ~= nil,
                      'criterion components should have a descriptive tostring' ..
                      ' beginning with the classname')
    end
end

function criterion_ignore_test.Parallel()
  local input = {torch.rand(2,10), torch.randn(2,10)}
  local target = {torch.IntTensor{1,8}, torch.randn(2,10)}
  local nll = nn.ClassNLLCriterion()
  local mse = nn.MSECriterion()
  local pc = nn.ParallelIgnore()
  pc:add(nll, 0.5):add(mse)
  local output = pc:forward(input, target)
  local output2 = nll:forward(input[1], target[1])/2 + mse:forward(input[2], target[2])
  mytester:assert(math.abs(output2 - output) < 0.00001, "ParallelCriterionIgnoreLabel forward error")
  local gradInput2 = {nll:backward(input[1], target[1]):clone():div(2), mse:backward(input[2], target[2])}
  local gradInput = pc:backward(input, target)
  mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.000001, "ParallelCriterionIgnoreLabel backward error 1")
  mytester:assertTensorEq(gradInput[2], gradInput2[2], 0.000001, "ParallelCriterionIgnoreLabel backward error 2")

  -- test type
  pc:float()
  gradInput[1], gradInput[2] = gradInput[1]:clone(), gradInput[2]:clone()
  local input3 = {input[1]:float(), input[2]:float()}
  local target3 = {target[1]:float(), target[2]:float()}
  local output3 = pc:forward(input3, target3)
  local gradInput3 = pc:backward(input3, target3)
  mytester:assert(math.abs(output3 - output) < 0.00001, "ParallelCriterionIgnoreLabel forward error type")
  mytester:assertTensorEq(gradInput[1]:float(), gradInput3[1], 0.000001, "ParallelCriterionIgnoreLabel backward error 1 type")
  mytester:assertTensorEq(gradInput[2]:float(), gradInput3[2], 0.000001, "ParallelCriterionIgnoreLabel backward error 2 type")

  -- test repeatTarget
  local input1 = {torch.rand(2,10), torch.randn(2,10)}
  local target1 = torch.randn(2,10)
  local mse1 = nn.MSECriterion()
  local pc1 = nn.ParallelIgnore(true):
    add(mse1, 0.5):
    add(mse1:clone())
  local output1_1 = pc1:forward(input1, target1)
  local output1_2 = mse1:forward(input1[1], target1)/2 + mse1:forward(input1[2], target1)
  mytester:assert(math.abs(output1_2 - output1_1) < 0.00001, "ParallelCriterionIgnoreLabel repeatTarget forward error")
  local gradInput1_1 = pc1:backward(input1, target1)
  local gradInput1_2 = {mse1:backward(input1[1], target1):clone():div(2), mse1:backward(input1[2], target1)}
  mytester:assertTensorEq(gradInput1_1[1], gradInput1_2[1], 0.000001, "ParallelCriterionIgnoreLabel repeatTarget backward error 1")
  mytester:assertTensorEq(gradInput1_1[2], gradInput1_2[2], 0.000001, "ParallelCriterionIgnoreLabel repeatTarget backward error 2")

  -- table input
  local input2 = {torch.randn(2,10), {torch.rand(2,10), torch.randn(2,10)}}
  local target2 = {torch.IntTensor{2,5}, {torch.IntTensor{1,8}, torch.randn(2,10)}}
  local nll2_2 = nn.ClassNLLCriterion()
  local nll2_1 = nn.ClassNLLCriterion()
  local mse2 = nn.MSECriterion()
  local pc2_1 = nn.ParallelIgnore():add(nll2_1, 0.5):add(mse2)
  local pc2_2 = nn.ParallelIgnore():add(nll2_2, 0.4):add(pc2_1)
  local output2_1 = pc2_2:forward(input2, target2)
  local output2_2 = nll2_2:forward(input2[1], target2[1])*0.4 + nll2_1:forward(input2[2][1], target2[2][1])/2 + mse2:forward(input2[2][2], target2[2][2])
  mytester:assert(math.abs(output2_2 - output2_1) < 0.00001, "ParallelCriterionIgnoreLabel table forward error")
  local gradInput2_2 = {
     nll2_2:backward(input2[1], target2[1]):clone():mul(0.4),
    {nll2_1:backward(input2[2][2], target2[2][1]):clone():div(2), mse2:backward(input2[2][2], target2[2][2])}
  }
  local gradInput2_1 = pc2_2:backward(input2, target2)
  mytester:assertTensorEq(gradInput2_1[1], gradInput2_2[1], 0.000001, "ParallelCriterionIgnoreLabel table backward error 1")
  mytester:assertTensorEq(gradInput2_1[2][1], gradInput2_2[2][1], 0.000001, "ParallelCriterionIgnoreLabel table backward error 2")
  mytester:assertTensorEq(gradInput2_1[2][2], gradInput2_2[2][2], 0.000001, "ParallelCriterionIgnoreLabel table backward error 3")


  -- Test ignore cases
  local input3 = {torch.rand(2,10), torch.rand(1,10)}
  local target3 = {torch.IntTensor{1,8}, torch.IntTensor{1}}
  local nll3_1 = nn.ClassNLLCriterion()
  local nll3_2 = nn.ClassNLLCriterion()
  local pc3 = nn.ParallelIgnore():
    add{
      criterion = nll3_1,
      weight = 0.5}:
    add{
     criterion = nll3_2,
     weight = 1,
     ignore = 0}
  local output3_1 = pc3:forward(input3, target3)
  local output3_2 = nll3_1:forward(input3[1], target3[1])/2 + nll3_2:forward(input3[2], target3[2])
  mytester:assert(math.abs(output3_2 - output3_1) < 0.00001, "ParallelCriterionIgnoreLabel forward error")
  local gradInput3_2 = {nll3_1:backward(input3[1], target3[1]):clone():div(2), nll3_2:backward(input3[2], target3[2])}
  local gradInput3_1 = pc3:backward(input3, target3)
  mytester:assertTensorEq(gradInput3_1[1], gradInput3_2[1], 0.000001, "ParallelCriterionIgnoreLabel backward error 4")
  mytester:assertTensorEq(gradInput3_1[2], gradInput3_2[2], 0.000001, "ParallelCriterionIgnoreLabel backward error 5")

  -- try to ignore one side
  target3[2] = 0
  local output3_3 = pc3:forward(input3, target3)
  local output3_4 = nll3_1:forward(input3[1], target3[1])/2
  mytester:assert(math.abs(output3_4 - output3_3) < 0.00001, "ParallelCriterionIgnoreLabel ignore doesn't work out as planned")

  local gradInput3_4 = nll3_1:backward(input3[1], target3[1]):clone():div(2)
  local gradInput3_3 = pc3:backward(input3, target3)
  mytester:assertTensorEq(gradInput3_3[1], gradInput3_4, 0.000001, "ParallelCriterionIgnoreLabel backward error with missing 6")
  mytester:assert(torch.sum(gradInput3_3[2]) < 0.000001, "ParallelCriterionIgnoreLabel backward error with missing fails to ignore weights 7")
end

mytester:add(criterion_ignore_test)

mytester:run()
