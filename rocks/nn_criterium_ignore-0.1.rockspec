package = "nn_criterium_ignore"
 version = "0.1"
 source = {
    url = "https://github.com/gforge/nn_criterium_ignore.git"
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
    url = "https://github.com/gforge/nn_criterium_ignore",
    license = "MIT/X11"
 }
 dependencies = {
    "lua ~> 5.1"
    "nn >= scm-1"
 }
 build = {
    -- We'll start here.
 }
