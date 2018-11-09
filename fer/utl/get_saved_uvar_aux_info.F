      SUBROUTINE GET_SAVED_UVAR_AUX_INFO( uvar, dset, aux_cat, aux_var, status)
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

* get the auxiliary variable info that has been associated with a saved
* uvar grid/dset pair.

* V699 5/16 *sh* new routine created to avoid the 2d uvar_grid array
* V71 10/16 *acm* Send back a status flag.

* arguments:
*     uvar - the Ferret user variable number (input)
*     aux_cat - array of nferdims category's for auxiliary regrid variables
*     aux_var - array of nferdims variables for auxiliary regridding
*     dset    - the dset in which this uvar is being evaluated (see below)

        IMPLICIT NONE
        include 'ferret.parm'
        include 'errmsg.parm'
        include 'tmap_dims.parm'
        include 'xvariables.cmn'

* calling argument declarations
        INTEGER uvar, aux_cat(nferdims), aux_var(nferdims), dset, status

* internal variable declarations
        INTEGER NCF_GET_UVAR_AUX_INFO, c_list_dset, context_dset

* c_list_dset  - dset value relevant in C LIST structures of NCF_Util.c
* context_dset - dset for which grid of this uvar has been determined

        IF ( uvar_dset(uvar) .GT. pdset_irrelevant) THEN
* ... LET/D= variable
          c_list_dset  = uvar_dset(uvar) ! the dset that owns this LET/D
          context_dset = uvar_dset(uvar) ! only one dset is possible
        ELSE
* ... global LET variable
          c_list_dset  = pdset_uvars  ! fake dataset of global uvars
          context_dset = dset
        ENDIF

* store the grid
        status = NCF_GET_UVAR_AUX_INFO( c_list_dset, uvar,
     .                          context_dset, aux_cat, aux_var )

        RETURN
        END
