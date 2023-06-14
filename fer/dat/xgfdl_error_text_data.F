	BLOCK DATA XERRMSG_TEXT_DATA

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* error message strings
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 -  3/31/86
* revision 0.1 --> 3/18/87+ - added several gerrs
* revision 1.0  -  5/14/87 - added last error and error codes
* revision 1.1  -  9/25/87 - changed relative coords error message
* revision 1.2  -  2/15/88 - data_set_not_set --> grid_definition 
* revision 1.3  -  3/2/88  - contour_line --> no_range
* revision 1.4  -  3/10/88 - k_depth --> perm_var
* revision 1.5  -  4/6/88  - cos_innac --> descriptor
* revision 1.6 -   8/4/88  - changed insuff memory text
* revision 1.7 -   1/3/89  - too_many_dsets --> regrid
* revision 1.8 -  1/12/89  - changed error texts to be compatible with
*			     ERRMSG prefixing "ERROR:"
* V200: 10/16/89 - added ferr_ODR_error for Panasonic laser disk
*	  2/7/90 - changed text of unknown_arg
*        23/9/91 - unused ferr_gks --> ferr_not_implemented
* V230:   8/5/92 - "ignored" removed from text of ferr_too_many_args
*        11/9/92 - text of unknown variable mentions not in data set
* V530: *sh* 9/00 - added ferr_data_type
* V580 *acm* 11/04 - New code ferr_nomessge for use when user-given error  
*                    msg in coached string substitution
* V6.2 *acm*  2/09 - Change the text of insufficent memory error to 
*                    request exceeds memory setting
* V6.7 *acm*  1/11 - Fix ticket 1768: undefined messages for no-such-attribute
*                    and error in attribute caused nulls to be written to std out.
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V684 9/12 *acm*  - Add error code for file not found
*                    (error message for font files not found)
* V658+ *sh* 10/13 - added ferr_unknown_auxvar
* V72  *acm*  6/17 - add ferr_ppl_error to call errmsg routine for syntax
*                    errors calling PPLUS, see ticket 2543
* V76:  1/2020 -     Remove obsolete ferr_ODR_error 
** 10/2022 *acm* Code cleanup: lengths of strings not to overflow length of character variable

        IMPLICIT NONE
	include 'errmsg.parm'	! parameter definitions
	include 'xerrmsg_text.cmn'	! common declaration


* last error reported
	DATA last_error / ferr_ok /

* codes for "special" errors
	DATA special_err_code( ferr_ok        - gspecial ) / 'ok'	/,
     .	     special_err_code( ferr_erreq     - gspecial ) / 'erreq'	/,
     .	     special_err_code( ferr_interrupt - gspecial ) / 'interrupt'/

* codes for errors that are not reported explicitly by GFDL
	DATA noreport_err_code( ferr_TMAP_error - gnoreport ) / 'TMAP_error'/

* codes and text for normal error messages
	DATA err_text( ferr_insuff_memory - greport )
     .		/ 'request exceeds memory setting'/
	DATA report_err_code( ferr_insuff_memory - greport )
     .		/ 'insuff_memory'/

	DATA err_text( ferr_too_many_vars - greport )
     .		/ 'too many variables to store in memory'	/
	DATA report_err_code( ferr_too_many_vars - greport )
     .		/ 'too_many_vars'/

	DATA err_text( ferr_perm_var	  - greport )		! unused
     .		/ 'data flagged as permanent'		/
	DATA report_err_code( ferr_perm_var	  - greport )
     .		/ 'perm_var' /

	DATA err_text( ferr_syntax	  - greport )
     .		/ 'command syntax'			/
	DATA report_err_code( ferr_syntax	  - greport )
     .		/ 'syntax' /

	DATA err_text( ferr_unknown_qualifier - greport )
     .		/ 'unknown command qualifier'			/
	DATA report_err_code( ferr_unknown_qualifier- greport )
     .		/ 'unknown_qualifie' /

	DATA err_text( ferr_unknown_variable - greport )
     .		/ 'variable unknown or not in data set'		/

	DATA report_err_code( ferr_unknown_variable - greport )
     .		/ 'unknown_variable' /

	DATA err_text( ferr_invalid_command  - greport )
     .		/ 'invalid command'			/
	DATA report_err_code( ferr_invalid_command  - greport )
     .		/ 'invalid_command' /

	DATA err_text( ferr_regrid - greport )
     .		/ 'regridding'				/
	DATA report_err_code( ferr_regrid - greport )
     .		/ 'regridding' /

	DATA err_text( ferr_cmnd_too_complex - greport )
     .		/ 'command too complex'				/
	DATA report_err_code( ferr_cmnd_too_complex - greport )
     .		/ 'cmnd_too_complex' /

	DATA err_text( ferr_unknown_data_set - greport )
     .		/ 'unknown data set'				/
	DATA report_err_code( ferr_unknown_data_set - greport )
     .		/ 'unknown_data_set' /

	DATA err_text( ferr_too_many_args - greport )
     .		/ 'too many arguments'/
	DATA report_err_code( ferr_too_many_args - greport )
     .		/ 'too_many_args' /

	DATA err_text( ferr_not_implemented	- greport )
     .		/ 'feature not implemented'			/
	DATA report_err_code( ferr_not_implemented - greport )
     .		/ 'not_implemented' /

	DATA err_text( ferr_invalid_subcmnd - greport )
     .		/ 'invalid subcommand'				/
	DATA report_err_code( ferr_invalid_subcmnd - greport )
     .		/ 'invalid_subcmnd' /

	DATA err_text( ferr_relative_coord - greport )
     .		/ 'invalid use delta region'			/
	DATA report_err_code( ferr_relative_coord - greport )
     .		/ 'relative_coord' /

	DATA err_text( ferr_unknown_arg - greport )
     .		/ 'unknown or invalid argument'				/
	DATA report_err_code( ferr_unknown_arg - greport )
     .		/ 'unknown_arg' /

	DATA err_text( ferr_dim_underspec - greport )
     .		/ 'dimensions improperly specified'		/
	DATA report_err_code( ferr_dim_underspec - greport )
     .		/ 'dim_underspec' /

	DATA err_text( ferr_grid_definition - greport )
     .		/ 'improper grid or axis definition'		/
	DATA report_err_code( ferr_grid_definition- greport )
     .		/ 'grid_definition' /

	DATA err_text( ferr_internal - greport )
     .		/ 'internal program problem'			/
	DATA report_err_code( ferr_internal - greport )
     .		/ 'internal' /

	DATA err_text( ferr_line_too_long - greport )
     .		/ 'line too long'				/
	DATA report_err_code( ferr_line_too_long - greport )
     .		/ 'line_too_long' /

	DATA err_text( ferr_inconsist_plane - greport )
     .		/ 'inconsistent sizes of data regions'		/
	DATA report_err_code( ferr_inconsist_plane - greport )
     .		/ 'inconsist_plane' /

	DATA err_text( ferr_inconsist_grid - greport )
     .		/ 'inconsistent data grids'			/
	DATA report_err_code( ferr_inconsist_grid - greport )
     .		/ 'inconsist_grid' /

	DATA err_text( ferr_expr_too_complex - greport )
     .		/ 'expression too complex'			/
	DATA report_err_code( ferr_expr_too_complex - greport )
     .		/ 'expr_too_complex' /

	DATA err_text( ferr_stack_ovfl - greport )
     .		/ 'stack overflow'				/
	DATA report_err_code( ferr_stack_ovfl - greport )
     .		/ 'stack_ovfl' /

	DATA err_text( ferr_stack_undfl - greport )
     .		/ 'stack underflow'				/
	DATA report_err_code( ferr_stack_undfl - greport )
     .		/ 'stack_undfl' /

	DATA err_text( ferr_out_of_range - greport )
     .		/ 'value out of legal range'			/
	DATA report_err_code( ferr_out_of_range - greport )
     .		/ 'out_of_range' /

	DATA err_text( ferr_prog_limit - greport )
     .		/ 'a program limit has been reached'		/
	DATA report_err_code( ferr_prog_limit - greport )
     .		/ 'prog_limit' /

	DATA err_text( ferr_unknown_grid - greport )
     .		/ 'unknown defining grid'			/
	DATA report_err_code( ferr_unknown_grid - greport )
     .		/ 'unknown_grid' /

	DATA err_text( ferr_no_range - greport )
     .		/ 'all data have same value'			/
	DATA report_err_code( ferr_no_range - greport )
     .		/ 'no_range' /

	DATA err_text( ferr_var_not_in_set - greport )
     .		/ 'required variable not in data set'		/
	DATA report_err_code( ferr_var_not_in_set - greport )
     .		/ 'var_not_in_set' /

	DATA err_text( ferr_unknown_file_type - greport )
     .		/ 'class of data set is unknown'		/
	DATA report_err_code( ferr_unknown_file_type - greport )
     .		/ 'unknown_file_typ' /

	DATA err_text( ferr_limits - greport )
     .		/ 'illegal limits'				/
	DATA report_err_code( ferr_limits - greport )
     .		/ 'limits' /

	DATA err_text( ferr_descriptor - greport )
     .		/ 'in descriptor file'	/
	DATA report_err_code( ferr_descriptor - greport )
     .		/ 'descriptor' /

	DATA err_text( ferr_bad_delta - greport )
     .		/ 'invalid delta increment'			/
	DATA report_err_code( ferr_bad_delta - greport )
     .		/ 'bad_delta' /

	DATA err_text( ferr_trans_nest - greport )
     .		/ 'illegal use of transformation'		/
	DATA report_err_code( ferr_trans_nest - greport )
     .		/ 'trans_nest' /

	DATA err_text( ferr_state_not_set - greport )
     .		/ 'required program command has not been given'	/
	DATA report_err_code( ferr_state_not_set - greport )
     .		/ 'state_not_set' /

	DATA err_text( ferr_unknown_command - greport )
     .		/ 'unknown command'				/
	DATA report_err_code( ferr_unknown_command - greport )
     .		/ 'unknown_command' /

        
	DATA err_text( ferr_ef_error - greport )
     .		/ ' '	/

	DATA err_text( ferr_data_type - greport )
     .		/ 'illegal data type (float,string,...) for operation'/
	DATA report_err_code( ferr_data_type - greport )
     .		/ 'data_type' /

	DATA err_text( ferr_unknown_attribute - greport )
     .		/ 'attribute undefined or invalid attribute name'/
	DATA report_err_code( ferr_unknown_attribute - greport )
     .		/ 'unknown_attribut' /

	DATA err_text( ferr_not_attribute - greport )
     .		/ 'error specifying attribute'/
	DATA report_err_code( ferr_not_attribute - greport )
     .		/ 'not_attribute' /

	DATA err_text( ferr_aggregate_error - greport )
     .		/ 'error defining aggregate dataset'/
	DATA report_err_code( ferr_not_attribute - greport )
     .		/ 'aggregate_error' /

	DATA err_text( ferr_file_not_found - greport )
     .		/ 'file undefined or not found'/
	DATA report_err_code( ferr_aggregate_error - greport )
     .		/ 'filefound_error' /

	DATA err_text( ferr_unknown_auxvar - greport )
     .		/ 'auxiliary variable unknown or not in data set' /
	DATA report_err_code( ferr_aggregate_error - greport )
     .		/ 'auxvar_error' /

	DATA err_text( ferr_ppl_error - greport )
     .		/ 'PPLUS command syntax error' /
	DATA report_err_code( ferr_ppl_error - greport )
     .		/ 'ppl_cmd_error' /

	DATA err_text( ferr_sys_return - greport )
     .		/ 'Shell command returned failure flag' /

	DATA err_text( ferr_nomessge - greport )
     .		/ ' '/

	DATA report_err_code( ferr_nomessge - greport )
     .		/ 'no_message' /

	END
