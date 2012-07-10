#! /bin/tcsh -f
#
# C-shell script to check for functions that need to be renamed
# for the 6D version
#

# list of functions that need to be renamed
set funclist = ""
set funclist = "$funclist ef_get_arg_subscripts"
set funclist = "$funclist ef_get_axis_info"
set funclist = "$funclist ef_get_res_subscripts"
set funclist = "$funclist ef_get_string_arg_element"
set funclist = "$funclist ef_get_string_arg_element_len"
set funclist = "$funclist ef_set_axis_influence"
set funclist = "$funclist ef_set_axis_inheritance"
set funclist = "$funclist ef_set_axis_reduction"
set funclist = "$funclist ef_set_piecemeal_ok"
set funclist = "$funclist ef_set_work_array_dims"
set funclist = "$funclist ef_set_work_array_lens"

# case-insensitive search for the above functions without _6d appended
foreach func ( ${funclist} )
   grep -i ${func} $argv | grep -i -v ${func}_6d
end

