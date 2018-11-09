      SUBROUTINE UPDATE_ATTRIBUTES (dset, vname, uvar, status)


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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP

* V600 6/05 *acm* Attribute control.
* V670  2/11 *acm*   Fixes for inheriting attributes.
* V698  2/16 *acm* For ticket 2352: variables defined with LET/D are
*                  added to that dataset in the attribute structure.
* V720 *acm* 5/17 Implement specific calls in place of CD_GET_VAR_INFO

* Get the attribute information from the linked-list structure that is needed 
* to do a plot.


        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
        include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xtext_info.cmn'
	include 'xdset_info.cmn_text'
        include 'xdset_info.cd_equiv'
	external xdset_info_data

* Argument declarations

      INTEGER uvar, dset
      CHARACTER*(*) vname

* Local variable declarations
      LOGICAL NC_GET_ATTRIB, got_it, do_warn, uvflag
      INTEGER TM_LENSTR1, varid,
     .        attlen, attoutflag, status, vlen, maxlen
      REAL tmp
      CHARACTER*128 buff

      vlen = TM_LENSTR1(vname)
      CALL CD_GET_VAR_ID (dset, vname, varid, status)
      IF (status .NE. ferr_ok) GOTO 1100

* get the needed info for this variable

      CALL CD_GET_VAR_UVFLAG (dset, varid, uvflag, status)
      IF (status .NE. ferr_ok) GOTO 1200

      do_warn = .TRUE.

* Get long_name, units, missing_value

      IF (dset .EQ. unspecified_int4) RETURN
      IF (dset .GT. pdset_irrelevant) THEN

         maxlen = 128
         got_it = NC_GET_ATTRIB( dset, varid, 'long_name',
     .                     .NOT.do_warn, vname(:vlen), maxlen, attlen, 
     .                     attoutflag, ds_var_title(uvar), tmp )

         maxlen = 128
         got_it = NC_GET_ATTRIB( dset, varid, 'long_name_mod',
     .                     .NOT.do_warn, vname(:vlen), maxlen, attlen, 
     .                     attoutflag, ds_var_titl_mod(uvar), tmp )

         maxlen = 64
         got_it = NC_GET_ATTRIB( dset, varid, 'units',
     .                     do_warn, vname(:vlen), maxlen, attlen, 
     .                     attoutflag, ds_var_units(uvar), tmp )

         maxlen = 1
         got_it = NC_GET_ATTRIB( dset, varid, 'missing_value',
     .                do_warn, vname(:vlen), maxlen, attlen,
     .                attoutflag, buff, ds_bad_flag(uvar) )

       ENDIF
* For user variable update these in uvar_title, etc.

       IF (dset.EQ.pdset_uvars .OR. uvflag) THEN
         maxlen = 128
         got_it = NC_GET_ATTRIB( dset, varid, 'long_name',
     .                     .NOT.do_warn, vname(:vlen), maxlen, attlen, 
     .                     attoutflag, uvar_title(uvar), tmp )

         maxlen = 64
         got_it = NC_GET_ATTRIB( dset, varid, 'units',
     .                     do_warn, vname(:vlen), maxlen, attlen, 
     .                     attoutflag, uvar_units(uvar), tmp )

         maxlen = 1
         got_it = NC_GET_ATTRIB( dset, varid, 'missing_value',
     .                do_warn, vname(:vlen), maxlen, attlen,
     .                attoutflag, buff, uvar_bad_data(uvar) )

      ENDIF

* successful completion - variable attributes updated
 1000 status = ferr_ok
 5000 RETURN
 1100 CALL ERRMSG
     .       ( ferr_unknown_variable, status, vname, *5000 )
 1200 CALL ERRMSG
     .       ( ferr_internal, status, 'varinfo: '//vname(:vlen), *5000 )
      END
