local torch = require 'torch'

criterion_ignore = {}

-- Add functions
-- todo: Extend to single criterions
criterion_ignore.Parallel = require 'criterion_ignore.Parallel'

return criterion_ignore