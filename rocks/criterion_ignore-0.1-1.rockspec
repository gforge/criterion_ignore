package = "criterion_ignore"
version = "0.1-1"
source = {
  url = "git://github.com/gforge/nn_criterium_ignore"
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
  homepage = "https://github.com/gforge/criterium_ignore",
  license = "MIT/X11",
  maintainer = "Max Gordon <max@gforge.se>"
}
dependencies = {
  "lua ~> 5.1",
  "nn >= scm-1",
  "torch >= 7.0"
}
build = {
  type = 'builtin',
  modules = {
    ["criterion_ignore.init"] = 'init.lua',
    ["criterion_ignore.Parallel"] = 'src/ParallelCriterionIgnoreLabel.lua'
  }
}
