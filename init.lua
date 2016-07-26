require('torch')

local argcheck_file = paths.thisfile():gsub("init.lua$", "argcheck.lua")
assert(loadfile(argcheck_file),
"Couldn't load " .. argcheck_file)()

local criterion_ignore = {}

-- Add functions
-- todo: Extend to single criterions
local main_file = paths.thisfile():gsub("init.lua$", "src/ParallelIgnore.lua")
  criterion_ignore.ParallelIgnore = assert(
  loadfile(main_file),
  "Couldn't load " .. main_file)()

return criterion_ignore
