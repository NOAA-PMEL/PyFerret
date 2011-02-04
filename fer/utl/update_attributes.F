      SUBROUTINE UPDATE_ATTRIBUTES (dset, uvar)


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

* Get the attribute information from the linked-list structure that is needed 
* to do a plot.


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

* Local variable declarations
      LOGICAL NC_GET_ATTRIB, coordvar, got_it, do_warn
      INTEGER TM_LENSTR1, varid, vartyp, nvdim, vdims(8),
     .        nvatts, attlen, attoutflag, status, vlen, maxlen, 
     .        all_outflag
      REAL tmp
      CHARACTER*128 vname, buff

      varid = cd_varid(uvar)
      IF (dset .EQ. -1) varid = uvar

* get the vital statistics for this variable

      CALL CD_GET_VAR_INFO(dset, varid, vname, vartyp, 
     .      nvdim, vdims, nvatts, coordvar, all_outflag, status)
      vlen = TM_LENSTR1(vname)

      do_warn = .TRUE.

* Get long_name, units, missing_value

      IF (dset .EQ. unspecified_int4) RETURN
      IF (dset .GE. 1) THEN
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

* For user variable update these in uvar_title, etc.

       ELSE IF (dset .EQ. -1) THEN
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
      RETURN
      END
