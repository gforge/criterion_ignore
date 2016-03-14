-- you can easily test specific units like this:
-- th -lnn -e "nn.test{'ParrallelCriterionIgnoreLabel'}"

local mytester = torch.Tester()
local jac

local precision = 1e-5
local expprecision = 1e-4

local nn_criterium_ignore_test = torch.TestSuite()


--[[ Generate tests to exercise the tostring component of modules. ]]
local tostringTestModules = {
    nnLinear = nn.Linear(1, 2),
    nnReshape = nn.Reshape(10),
    nnSpatialZeroPadding = nn.SpatialZeroPadding(1, 1, 1, 1)}
for test_name, component in pairs(tostringTestModules) do
  nn_criterium_ignore_test['tostring' .. test_name] =
    function ()
      mytester:assert(tostring(component):find(
                         torch.type(component) .. '(', 1, true) ~= nil,
                      'nn components should have a descriptive tostring' ..
                      ' beginning with the classname')
    end
end

function nn_criterium_ignore_test.ParrallelCriterionIgnoreLabel()
   local input = {torch.rand(2,10), torch.randn(2,10)}
   local target = {torch.IntTensor{1,8}, torch.randn(2,10)}
   local nll = nn.ClassNLLCriterion()
   local mse = nn.MSECriterion()
   local pc = nn.ParrallelCriterionIgnoreLabel():add(nll, 0.5):add(mse)
   local output = pc:forward(input, target)
   local output2 = nll:forward(input[1], target[1])/2 + mse:forward(input[2], target[2])
   mytester:assert(math.abs(output2 - output) < 0.00001, "ParrallelCriterionIgnoreLabel forward error")
   local gradInput2 = {nll:backward(input[1], target[1]):clone():div(2), mse:backward(input[2], target[2])}
   local gradInput = pc:backward(input, target)
   mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.000001, "ParrallelCriterionIgnoreLabel backward error 1")
   mytester:assertTensorEq(gradInput[2], gradInput2[2], 0.000001, "ParrallelCriterionIgnoreLabel backward error 2")

   -- test type
   pc:float()
   gradInput[1], gradInput[2] = gradInput[1]:clone(), gradInput[2]:clone()
   local input3 = {input[1]:float(), input[2]:float()}
   local target3 = {target[1]:float(), target[2]:float()}
   local output3 = pc:forward(input3, target3)
   local gradInput3 = pc:backward(input3, target3)
   mytester:assert(math.abs(output3 - output) < 0.00001, "ParrallelCriterionIgnoreLabel forward error type")
   mytester:assertTensorEq(gradInput[1]:float(), gradInput3[1], 0.000001, "ParrallelCriterionIgnoreLabel backward error 1 type")
   mytester:assertTensorEq(gradInput[2]:float(), gradInput3[2], 0.000001, "ParrallelCriterionIgnoreLabel backward error 2 type")

   -- test repeatTarget
   local input = {torch.rand(2,10), torch.randn(2,10)}
   local target = torch.randn(2,10)
   local mse = nn.MSECriterion()
   local pc = nn.ParrallelCriterionIgnoreLabel(true):add(mse, 0.5):add(mse:clone())
   local output = pc:forward(input, target)
   local output2 = mse:forward(input[1], target)/2 + mse:forward(input[2], target)
   mytester:assert(math.abs(output2 - output) < 0.00001, "ParrallelCriterionIgnoreLabel repeatTarget forward error")
   local gradInput = pc:backward(input, target)
   local gradInput2 = {mse:backward(input[1], target):clone():div(2), mse:backward(input[2], target)}
   mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.000001, "ParrallelCriterionIgnoreLabel repeatTarget backward error 1")
   mytester:assertTensorEq(gradInput[2], gradInput2[2], 0.000001, "ParrallelCriterionIgnoreLabel repeatTarget backward error 2")

   -- table input
   local input = {torch.randn(2,10), {torch.rand(2,10), torch.randn(2,10)}}
   local target = {torch.IntTensor{2,5}, {torch.IntTensor{1,8}, torch.randn(2,10)}}
   local nll2 = nn.ClassNLLCriterion()
   local nll = nn.ClassNLLCriterion()
   local mse = nn.MSECriterion()
   local pc = nn.ParrallelCriterionIgnoreLabel():add(nll, 0.5):add(mse)
   local pc2 = nn.ParrallelCriterionIgnoreLabel():add(nll2, 0.4):add(pc)
   local output = pc2:forward(input, target)
   local output2 = nll2:forward(input[1], target[1])*0.4 + nll:forward(input[2][1], target[2][1])/2 + mse:forward(input[2][2], target[2][2])
   mytester:assert(math.abs(output2 - output) < 0.00001, "ParrallelCriterionIgnoreLabel table forward error")
   local gradInput2 = {
       nll2:backward(input[1], target[1]):clone():mul(0.4),
      {nll:backward(input[2][2], target[2][1]):clone():div(2), mse:backward(input[2][2], target[2][2])}
   }
   local gradInput = pc2:backward(input, target)
   mytester:assertTensorEq(gradInput[1], gradInput2[1], 0.000001, "ParrallelCriterionIgnoreLabel table backward error 1")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2][1], 0.000001, "ParrallelCriterionIgnoreLabel table backward error 2")
   mytester:assertTensorEq(gradInput[2][2], gradInput2[2][2], 0.000001, "ParrallelCriterionIgnoreLabel table backward error 3")
end

mytester:add(nn_criterium_ignore_test)

if not nn_criterium_ignore then
   require 'nn_criterium_ignore'
   mytester:run()
else
   function nn.test(tests, seed)
      -- randomize stuff
      local seed = seed or os.time()
      print('Seed: ', seed)
      math.randomseed(seed)
      torch.manualSeed(seed)
      mytester:run(tests)
      return mytester
   end
end