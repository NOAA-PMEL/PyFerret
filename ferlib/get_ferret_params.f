*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY
*  SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.



*
* Returns a listing of ferret parameter values and appropriate names for these
* values.  The ordering of the values in these arrays is not specified beyond
* that a name is appropriate for the corresponding value.  The names are unique
* in this list and are null terminated.
*
      SUBROUTINE GET_FERRET_PARAMS(NAMES, VALUES, NUMVALS)

      IMPLICIT NONE
      INCLUDE 'errmsg.parm'
      INCLUDE 'EF_Util.parm'

      INTEGER MAXVALS
      CHARACTER*32 NAMES(64)
      INTEGER VALUES(64), NUMVALS

      NAMES(1) = 'FERR_OK' // CHAR(0)
      VALUES(1) = ferr_ok

      NAMES(2) = 'FERR_ERREQ' // CHAR(0)
      VALUES(2) = ferr_erreq

      NAMES(3) = 'FERR_INTERRUPT' // CHAR(0)
      VALUES(3) = ferr_interrupt

      NAMES(4) = 'FERR_TMAP_ERROR' // CHAR(0)
      VALUES(4) = ferr_tmap_error

      NAMES(5) = 'FERR_ODR_ERROR' // CHAR(0)
      VALUES(5) = ferr_odr_error

      NAMES(6) = 'FERR_SILENT_ERROR' // CHAR(0)
      VALUES(6) = ferr_silent

      NAMES(7) = 'FERR_INSUFF_MEMORY' // CHAR(0)
      VALUES(7) = ferr_insuff_memory

      NAMES(8) = 'FERR_TOO_MANY_VARS' // CHAR(0)
      VALUES(8) = ferr_too_many_vars

      NAMES(9) = 'FERR_DEL_PERM_VAR' // CHAR(0)
      VALUES(9) = ferr_perm_var

      NAMES(10) = 'FERR_SYNTAX_ERROR' // CHAR(0)
      VALUES(10) = ferr_syntax

      NAMES(11) = 'FERR_UNKNOWN_QUALIFIER' // CHAR(0)
      VALUES(11) = ferr_unknown_qualifier

      NAMES(12) = 'FERR_UNKNOWN_VARIABLE' // CHAR(0)
      VALUES(12) = ferr_unknown_variable

      NAMES(13) = 'FERR_INVALID_COMMAND' // CHAR(0)
      VALUES(13) = ferr_invalid_command

      NAMES(14) = 'FERR_REGRID_ERROR' // CHAR(0)
      VALUES(14) = ferr_regrid

      NAMES(15) = 'FERR_CMND_TOO_COMPLEX' // CHAR(0)
      VALUES(15) = ferr_cmnd_too_complex

      NAMES(16) = 'FERR_UNKNOWN_DATA_SET' // CHAR(0)
      VALUES(16) = ferr_unknown_data_set

      NAMES(17) = 'FERR_TOO_MANY_ARGS' // CHAR(0)
      VALUES(17) = ferr_too_many_args

      NAMES(18) = 'FERR_NOT_IMPLEMENTED' // CHAR(0)
      VALUES(18) = ferr_not_implemented

      NAMES(19) = 'FERR_INVALID_SUBCMND' // CHAR(0)
      VALUES(19) = ferr_invalid_subcmnd

      NAMES(20) = 'FERR_RELATIVE_COORD_ERROR' // CHAR(0)
      VALUES(20) = ferr_relative_coord

      NAMES(21) = 'FERR_UNKNOWN_ARG' // CHAR(0)
      VALUES(21) = ferr_unknown_arg

      NAMES(22) = 'FERR_DIM_UNDERSPEC' // CHAR(0)
      VALUES(22) = ferr_dim_underspec

      NAMES(23) = 'FERR_GRID_DEF_ERROR' // CHAR(0)
      VALUES(23) = ferr_grid_definition

      NAMES(24) = 'FERR_INTERNAL_ERROR' // CHAR(0)
      VALUES(24) = ferr_internal

      NAMES(25) = 'FERR_LINE_TOO_LONG' // CHAR(0)
      VALUES(25) = ferr_line_too_long

      NAMES(26) = 'FERR_INCONSIST_PLANE' // CHAR(0)
      VALUES(26) = ferr_inconsist_plane

      NAMES(27) = 'FERR_INCONSIST_GRID' // CHAR(0)
      VALUES(27) = ferr_inconsist_grid

      NAMES(28) = 'FERR_EXPR_TOO_COMPLEX' // CHAR(0)
      VALUES(28) = ferr_expr_too_complex

      NAMES(29) = 'FERR_STACK_OVERFLOW' // CHAR(0)
      VALUES(29) = ferr_stack_ovfl

      NAMES(30) = 'FERR_STACK_UNDERFLOW' // CHAR(0)
      VALUES(30) = ferr_stack_undfl

      NAMES(31) = 'FERR_OUT_OF_RANGE' // CHAR(0)
      VALUES(31) = ferr_out_of_range

      NAMES(32) = 'FERR_PROG_LIMIT' // CHAR(0)
      VALUES(32) = ferr_prog_limit

      NAMES(33) = 'FERR_UNKNOWN_GRID' // CHAR(0)
      VALUES(33) = ferr_unknown_grid

      NAMES(34) = 'FERR_NO_RANGE' // CHAR(0)
      VALUES(34) = ferr_no_range

      NAMES(35) = 'FERR_VAR_NOT_IN_SET' // CHAR(0)
      VALUES(35) = ferr_var_not_in_set

      NAMES(36) = 'FERR_UNKNOWN_FILE_TYPE' // CHAR(0)
      VALUES(36) = ferr_unknown_file_type

      NAMES(37) = 'FERR_LIMITS_ERROR' // CHAR(0)
      VALUES(37) = ferr_limits

      NAMES(38) = 'FERR_DESCRIPTOR_ERROR' // CHAR(0)
      VALUES(38) = ferr_descriptor

      NAMES(39) = 'FERR_BAD_DELTA' // CHAR(0)
      VALUES(39) = ferr_bad_delta

      NAMES(40) = 'FERR_TRANSFORM_ERROR' // CHAR(0)
      VALUES(40) = ferr_trans_nest

      NAMES(41) = 'FERR_STATE_NOT_SET' // CHAR(0)
      VALUES(41) = ferr_state_not_set

      NAMES(42) = 'FERR_UNKNOWN_COMMAND' // CHAR(0)
      VALUES(42) = ferr_unknown_command

      NAMES(43) = 'FERR_EF_ERROR' // CHAR(0)
      VALUES(43) = ferr_ef_error

      NAMES(44) = 'FERR_DATA_TYPE_ERROR' // CHAR(0)
      VALUES(44) = ferr_data_type

      NAMES(45) = 'FERR_NO_COACH_MESSAGE' // CHAR(0)
      VALUES(45) = ferr_nomessge

      NAMES(46) = 'FERR_UNKNOWN_ATTRIBUTE' // CHAR(0)
      VALUES(46) = ferr_unknown_attribute

      NAMES(47) = 'FERR_NOT_ATTRIBUTE' // CHAR(0)
      VALUES(47) = ferr_not_attribute

      NAMES(48) = 'AXIS_CUSTOM' // CHAR(0)
      VALUES(48) = custom

      NAMES(49) = 'AXIS_IMPLIED_BY_ARGS' // CHAR(0)
      VALUES(49) = implied_by_args

      NAMES(50) = 'AXIS_DOES_NOT_EXIST' // CHAR(0)
      VALUES(50) = normal

      NAMES(51) = 'AXIS_ABSTRACT' // CHAR(0)
      VALUES(51) = abstract

      NAMES(52) = 'AXIS_REDUCED' // CHAR(0)
      VALUES(52) = reduced

      NUMVALS = 52
      RETURN
      END

