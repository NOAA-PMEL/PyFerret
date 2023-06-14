        SUBROUTINE save_scriptfile_name(name, clen, status)

*
*
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
*        Ansley Manke PMEL/TMAP
*        8/2003  (based on save_metafile_name)
**       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)

* Put script file name into common, for running with the comand line option
*  ferret -script 

        IMPLICIT NONE
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'errmsg.parm'
        INCLUDE 'tmap_errors.parm'
        INCLUDE 'xprog_state.cmn'

* calling argument declarations:
        INTEGER clen, status
        INTEGER*1 name(clen)

* Local variable declarations

        CHARACTER TM_INQ_PATH*2048, cbuff*2048, sname*2048

        LOGICAL TM_HAS_STRING
        INTEGER ok, TMAP_status

        CALL tm_ctof_strng(name, sname, clen)
        scriptfile_name(1:clen) = sname

! Save the state, running with -script switch

        its_script = .TRUE.

        IF (TM_HAS_STRING(sname, '.jnl')) THEN
           ok = 1
        ELSE 
           scriptfile_name(clen+1:clen+4) = '.jnl'
           clen = clen + 4
        ENDIF

! See if the file exists

        cbuff = TM_INQ_PATH (scriptfile_name(1:clen), 'FER_GO',
     .                       '.jnl', .TRUE., TMAP_status)

        IF ( TMAP_status .NE. merr_ok ) status = 0


 5000	RETURN
        END

