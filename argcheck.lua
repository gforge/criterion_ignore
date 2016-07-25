local env = require 'argcheck.env' -- retrieve argcheck environement

env.istype = function(obj, typename)
	local thtype = torch.type(obj)
	if (typename == "table|torch.*Tensor") then
		return thtype == "table" or
			env.istype(obj, "torch.*Tensor")
	end

	-- From the original argcheck env
	local thname = torch.typename(obj)
	if thname then
		-- __typename (see below) might be absent
		local match = thname:match(typename)
		if match and (match ~= typename or match == thname) then
			return true
		end
		local mt = torch.getmetatable(thname)
		while mt do
			if mt.__typename then
				match = mt.__typename:match(typename)
				if match and (match ~= typename or match == mt.__typename) then
					return true
				end
			end
			mt = getmetatable(mt)
		end
		return false
	end

	return type(obj) == typename
end
