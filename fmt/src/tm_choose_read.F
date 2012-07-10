	SUBROUTINE TM_CHOOSE_READ ( dset_num, varpos,
     .		a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .		a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .		use_lo, use_hi, stride,
     .		data_grid, file_num, status )
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
*
* Reads a N-D block of data (3D, 2D being degenerate cases w/ 1 dimension being
* 1 time series) from TMAP data sets.
*
* written 10/02/88 by M. Verschell for PMEL/TMAP
*                     
* revision 0.00 - 10/02/88 - initial incarnation
* revision 0.10 - 11/01/88 - name change from TM_READ, passed var position in
*			     common rather than variable name.
* revision 0.20 - 12/06/89 - changed arrays for dimensions to single variables
*   3/92 - added CDF file reads using CD_READ (calling arg list chgd 10/92)
*   3/93 - treat ECDF (EPIC) like CDF
* V500 5/99 *sh* - added strides to CD_READ call
* 4/03 *acm* - Use STR_SAME to compare strings
*       *acm* 3/12 removing use of old GT and TS data formats

* include files
        include 'tmap_dims.parm'                ! data set dimension
        include 'tmap_errors.parm'              ! err code parm defs
#include "gt_lib.parm"
        include 'xtm_grid.cmn_text'             ! grid info
        external xgt_grid_data
        include 'xdset_info.cmn_text'   ! data set info common
        external xdset_info_data
        include 'xdset_info.cd_equiv'


* arguments:
*	dset_num    - internal data set # (from TM_INIT_DSET)
*	varpos      - position in common of variable information
* 	actual_dims - "a_" full dimensions of data_grid array passed
*	use_dims    - indices of model data block desired
*	data_grid   - array containing model data (output)
*	file_num    - file number (output)
*	status      - error status upon completion (output)
*
	INTEGER		dset_num, varpos, file_num, status,
     .			a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .			a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .			use_lo(nferdims), use_hi(nferdims), stride(nferdims)
        REAL            data_grid(a_lo_x:a_hi_x,
     .                            a_lo_y:a_hi_y,
     .                            a_lo_z:a_hi_z,
     .                            a_lo_t:a_hi_t,
     .                            a_lo_e:a_hi_e,
     .                            a_lo_f:a_hi_f)
*
* local definitions
	INTEGER*4	STR_SAME, lcnt, lline, ivar, tcnt, pcnt
	INTEGER*4	tmp_lo(nferdims), tmp_hi(nferdims)
	REAL*8		tstep, TM_WORLD

        is_mc = .FALSE.

* Is it a MC data set?
  	IF (STR_SAME(ds_type(dset_num), '  MC') .EQ. 0) THEN
            is_mc = .TRUE.
	    CALL MC_READ  ( dset_num, varpos,
     .			a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .			a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .			use_lo, use_hi, stride,
     .			data_grid, file_num, status )
	    IF (status .NE. merr_ok) GOTO 9999
	ELSEIF (STR_SAME(ds_type(dset_num), 'CDF') .EQ. 0
     .    .OR.  STR_SAME(ds_type(dset_num), 'ECDF') .EQ. 0) THEN
* It's a CDF or an EPIC CDF data set
          CALL CD_READ( dset_num, varpos,
     .			a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .			a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .                  use_lo, use_hi, stride,
     .                  data_grid, cd_stepfile(dset_num),status)
	  IF (status .NE. merr_ok) GOTO 9999
        ELSE

          CALL TM_ERRMSG(merr_notsupport, status, 'TM_CHOOSE_READ',
     .                  dset_num, no_stepfile,
     .                  'Unknown data set type: '//ds_type(dset_num),
     .                  no_errstring, *9999)
	ENDIF
	status = merr_ok

 9999	RETURN
	END
