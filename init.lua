require('torch')

local argcheck_file = paths.thisfile():gsub("init.lua$", "argcheck.lua")
assert(loadfile(argcheck_file),
       "Couldn't load " .. argcheck_file)()

criterion_ignore = {}

-- Add functions
-- todo: Extend to single criterions
local main_file = paths.thisfile():gsub("init.lua$", "src/Parallel.lua")
criterion_ignore.Parallel = assert(
  loadfile(main_file),
  "Couldn't load " .. main_file)()

print("ASDAS")
return criterion_ignore
