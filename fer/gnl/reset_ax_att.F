      SUBROUTINE RESET_AX_ATT (name, new_units, new_t0,  
     .       new_cal_name, new_modulo, new_modulo_len, 
     .       direction, regular_flag )

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
*
*  Change attributes for an axis

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP
* 11/2006 
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
*                  Use pseudo-dataset of user-defined axes to keep track of attributes.

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include 'errmsg.parm'
        include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xrisc.cmn'
        include 'xdset_parms.cmn'
        include 'netcdf.inc'

* calling argument declarations:

      CHARACTER*(*) name, direction, new_units, new_t0, 
     .              new_cal_name
      REAL*8 new_modulo_len
      LOGICAL regular_flag, new_modulo

* local variable declarations: 
      
      INTEGER TM_LENSTR1, STR_SAME, 
     .        dset, varid, attlen, attype, attoutflag, 
     .        vartype, nvdims, nvatts, vdims(8), all_outflag,
     .        attid, len, maxlen, status
      REAL dummy
      CHARACTER buff*128, varname*128, attbuff*10, attname*128, dir*4
      LOGICAL NC_GET_ATTRIB, coordvar, do_warn, got_it

      dummy = 0.
      do_warn = .FALSE.
      maxlen = 128

c... loop over datasets
c    see if name exists as an axis name.

      DO 100 dset = -2, maxdsets
         IF (dset .EQ. -1 .OR. dset .EQ. 0) GOTO 100
         
         IF (dset .GT. 0) THEN
            IF ( ds_name(dset) .EQ. char_init256) GOTO 100
         ENDIF

         CALL CD_GET_VAR_ID (dset, name, varid, status)

         IF (status .EQ. ferr_ok) THEN  
! name is a variable; is it a coordinate variable of the dataset?

            CALL CD_GET_VAR_INFO (dset, varid, name, vartype, 
     .          nvdims, vdims, nvatts, coordvar, all_outflag, status) 
            IF (.NOT. coordvar) goto 100

         ELSE
	    GOTO 100
	 ENDIF

c ... If units attribute already exists, change it; otherwise add new

         IF (new_units .NE. " ") THEN
            attname = 'units'
            len = TM_LENSTR1(new_units)
            CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)

            IF (status .NE. ferr_ok) THEN
               attoutflag = 1
               CALL CD_PUT_NEW_ATTR (dset, varid, attname, NCCHAR, len,
     .         attoutflag, new_units, dummy, status)
            ELSE
               CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, attname, attype,
     .           attlen, attoutflag, status)
               CALL CD_REPLACE_ATTR (dset, varid, 
     .           attname, NCCHAR, len, new_units, dummy, status)
            ENDIF
        ENDIF

c ... If t0 attribute already exists, change it; otherwise add new

         IF (new_t0 .NE. " ") THEN
            attname = 'time_origin'
            len = TM_LENSTR1(new_t0)
            CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)

            IF (status .NE. ferr_ok) THEN
               attoutflag = 1
               CALL CD_PUT_NEW_ATTR (dset, varid, attname, NCCHAR, len,
     .           attoutflag, new_t0, dummy, status)
            ELSE
               CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, attname, attype,
     .           attlen, attoutflag, status)
               CALL CD_REPLACE_ATTR (dset, varid, 
     .           attname, attype, len, new_t0, dummy, status)
            ENDIF
         ENDIF

c ... direction; Change positive=up or down on Z axes; replace cartesian_axis 
c     or axis attribute only if already an attribute

         attname = 'positive'

         CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)

         IF (status .EQ. ferr_ok) THEN
            CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, attname,
     .        attype, attlen, attoutflag, status)

            IF (direction .EQ. 'DU') THEN
               dir = 'up'
               len = 2
            ELSE
               dir = 'down'
               len = 4
            ENDIF
            CALL CD_REPLACE_ATTR (dset, varid, attname, attype, 
     .        len, dir, dummy, status)
         ENDIF

         attname = 'cartesian_axis'
         len = 1

         dir = ' '
         IF (direction(1:1) .EQ. 'T') dir = 'T' 
         IF (direction .EQ. 'WE' .OR. 
     .    direction(1:1) .EQ. 'X') dir = 'X'
         IF (direction .EQ. 'SN' .OR. 
     .    direction(1:1) .EQ. 'Y') dir = 'Y'
         IF (direction .EQ. 'UD' .OR. 
     .    direction .EQ. 'DU' .OR.  
     .    direction(1:1) .EQ. 'Z') dir = 'Z'
         
         IF (dir .NE. ' ') THEN
            CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)

            IF (status .EQ. ferr_ok) THEN
               CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, attname, attype,
     .           attlen, attoutflag, status)
               CALL CD_REPLACE_ATTR (dset, varid, 
     .           attname, attype, len, dir, dummy, status)
            ENDIF

            attname = 'axis'

            CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)

            IF (status .EQ. ferr_ok) THEN
               CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, attname, attype,
     .           attlen, attoutflag, status)
               CALL CD_REPLACE_ATTR (dset, varid, 
     .           attname, attype, len, dir, dummy, status)
            ENDIF
         ENDIF

c ... If calendar attribute already exists, change it; otherwise add new

         IF (new_cal_name .NE. " ") THEN
            attname = 'calendar'
            attoutflag = 1
            IF ( (STR_SAME(new_cal_name(1:3), 'GRE') .EQ. 0) .OR.
     .           (STR_SAME(new_cal_name(1:3), 'STA') .EQ. 0) ) attoutflag = 0

            len = TM_LENSTR1(new_cal_name)
            CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)

            IF (status .NE. ferr_ok) THEN
               CALL CD_PUT_NEW_ATTR (dset, varid, attname, NCCHAR, len,
     .           attoutflag, new_cal_name, dummy, status)
            ELSE
               CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, attname, attype,
     .           attlen, attoutflag, status)
               CALL CD_REPLACE_ATTR (dset, varid, 
     .           attname, attype, len, new_cal_name, dummy, status)
            ENDIF
         ENDIF

c ... point_spacing; change only if it was already an attribute

         attname = 'point_spacing'
         attbuff = 'uneven'
         IF (regular_flag) attbuff = 'even'
         len = TM_LENSTR1(attbuff)

         CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)

         IF (status .EQ. ferr_ok) THEN
            CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, attname, attype,
     .        attlen, attoutflag, status)
            CALL CD_REPLACE_ATTR (dset, varid, 
     .        attname, attype, len, attbuff, dummy, status)
         ENDIF

c ... modulo; either write modulo = ' ' for modulo=true,
c     or write modulo length.

         attname = 'modulo'
         IF (new_modulo) THEN
         
            IF (new_modulo .OR. (new_modulo_len .GT. 0.D0)) THEN
               attbuff = ' '
               len = 1
               CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)

               IF (new_modulo_len .GT. 0) THEN  ! write the value of new_modulo_len
                  IF (status .NE. ferr_ok) THEN
                     attoutflag = 1
                     CALL CD_PUT_NEW_ATTR_DP (dset, varid, attname, NCDOUBLE, len,
     .               attoutflag, attbuff, new_modulo_len, status)
                  ELSE
                     CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, attname, attype,
     .               attlen, attoutflag, status)
                     CALL CD_REPLACE_ATTR_DP (dset, varid, 
     .               attname, attype, len, attbuff, new_modulo_len, status)
                  ENDIF

               ELSE  ! write string attribute modulo=' '
                  IF (status .NE. ferr_ok) THEN
                     attoutflag = 1
                     CALL CD_PUT_NEW_ATTR_DP (dset, varid, attname, NCCHAR, len,
     .               attoutflag, attbuff, new_modulo_len, status)
                  ELSE
                     CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, attname, attype,
     .               attlen, attoutflag, status)
                     CALL CD_REPLACE_ATTR (dset, varid, 
     .               attname, NCCHAR, len, attbuff, dummy, status)
                  ENDIF
               ENDIF

            ENDIF
         ELSE  ! if new_modulo is false then set attribute from axis
 
            CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)

            IF (status .EQ. ferr_ok) THEN
               CALL CD_DELETE_ATTRIBUTE (dset, varid, attname, status)
            ENDIF
         ENDIF

  100 CONTINUE   
      
      status = ferr_ok

      RETURN
      END
