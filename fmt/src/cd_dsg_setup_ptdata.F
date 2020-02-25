        SUBROUTINE CD_DSG_SETUP_PTDATA(dset, npts, stored_lm, stored_lm1, status)


*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*.
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
* Set up a synthesized DSG rowsize variable from a Point netCDF dataset
* RowSize of length npts with all values 1

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - SDIG

* argument definitions:
*       npts      - number of values to put into variable
*       idim      - orientation of the coordinate (pdsg_row_size for rowSize)
*       stored_lm - dynamic memory line number of the instance variable created 
*       status    - return status

* include files
        include 'tmap_errors.parm'
        include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xdset_info.cmn_text'
        external xdset_info_data
        include 'xdyn_linemem.cmn_text'
        include 'xtm_grid.cmn_text'
        include 'xunits.cmn_text'

* argument declarations
        INTEGER  dset, ivcd, npts, stored_lm, stored_lm1, status

* local variable declarations:
        INTEGER iaxis
        REAL    coord

* local parameter declaration
        INTEGER		i
        INTEGER*8	rqst_n
	CHARACTER	axname*9

* allocate memory for the synthesized Rowsize variable
        rqst_n = npts
        CALL TM_GET_LM_DYNMEM(rqst_n, stored_lm, status)
        IF (status .NE. merr_ok) GOTO 5000

        coord = 1.
	DO i = 1, npts
           CALL PUT_LINE_COORD ( dsg_linemem(stored_lm)%ptr, i, coord )   
	ENDDO

* *** create the instance (E) axis for the dataset
        axname = "features"
        iaxis = 0
        CALL TM_MAKE_BASIC_AXIS( axname, 1.0D0, 1.0D0, npts,
     .				    iaxis, status )
        IF ( status .NE. merr_ok ) GOTO 5000
        line_direction(iaxis) = axis_orients(E_dim)
        line_units(iaxis)    = un_name(pun_instance)
* ... save this axis to be used during grid creation
        dsg_instance_axis = iaxis

* flag that this is a DSG axis
	line_dsg_dset(iaxis) = dset

* Don't need to allocate memory for the synthesized Obs axis
* just use the basic-axis (?)
c        rqst_n = 1
c        CALL TM_GET_LM_DYNMEM(rqst_n, stored_lm1, status)
c        IF (status .NE. merr_ok) GOTO 5000

* *** create the nominal obs (X) axis for the dataset
        axname = "ferObs"
        iaxis = 0
        CALL TM_MAKE_BASIC_AXIS( axname, 1.0D0, 1.0D0, npts,
     .				    iaxis, status )
        IF ( status .NE. merr_ok ) GOTO 5000
        line_direction(iaxis) = axis_orients(X_dim)
        line_units(iaxis)    = 'count'
* ... save this axis to be used during grid creation
        dsg_obs_axis = iaxis

* flag that this is a DSG axis
	line_dsg_dset(iaxis) = dset


* successful completion
        RETURN

* error exit
 5000   RETURN

        END

