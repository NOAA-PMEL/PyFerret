      SUBROUTINE GET_SAVED_UVAR_GRID_LIST_LEN( uvar, ngrids )
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

* return the number of grids that have been saved for this uvar 
* (one per associated dataset)

* V699 5/16 *sh* new routine created to avoid the 2d uvar_grid array


        IMPLICIT NONE
        include 'ferret.parm'
        include 'tmap_dims.parm'
        include 'xvariables.cmn'

* calling argument declarations
        INTEGER uvar, ngrids

* internal variable declarations
        INTEGER NCF_GET_UVAR_GRID_LIST_LEN, status, c_list_dset

* c_list_dset  - dset value relevant in C LIST structures of NCF_Util.c

        IF ( uvar_dset(uvar) .GT. pdset_irrelevant) THEN
* ... LET/D= variable
          c_list_dset  = uvar_dset(uvar) ! the dset that owns this LET/D
        ELSE
* ... global LET variable
          c_list_dset  = pdset_uvars  ! fake dataset of global uvars
        ENDIF

* get the number of grids
        status = NCF_GET_UVAR_GRID_LIST_LEN( c_list_dset, uvar, ngrids)

* check status
        IF (status.EQ.atom_not_found) ngrids = 0

        RETURN
        END
