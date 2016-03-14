package = "nn_criterium_ignore"
 version = "0.1"
 source = {
    url = "..." -- We don't have one yet
 }
 description = {
    summary = "A parallel criterion with ignore label",
    detailed = [[
       An extension to the ParallelCriterion that allows you
       to add labels that will be ignored. For each :add(criterion)
       you can also specify an ignoreLabel that will cause the 
       sum to exclude any target that matches the label. Note
       that the ignore label has to be <= 0 in order to make sure that
       there is no unexpected conflict with classes.
    ]],
    homepage = "http://...", -- We don't have one yet
    license = "MIT/X11" -- or whatever you like
 }
 dependencies = {
    "lua ~> 5.1"
    -- If you depend on other rocks, add them here
 }
 build = {
    -- We'll start here.
 }
