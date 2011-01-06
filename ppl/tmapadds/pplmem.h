/* pplmem.h 
   Declarations for routines that allow dynamic PPLUS memory buffer 
   9/18/01 *acm*

/*
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

/* Easier way of handling FORTRAN calls with underscore/no underscore */
#ifndef FORTRAN
#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif
#endif


void FORTRAN(pplcmd_c)(int *, int *, int *);
void FORTRAN(pplcmd_f)(int *, int *, int *, float * );

void FORTRAN(pplldx_envelope)(int *, float *, float *, int *, 
                       char *, char *, float *, int *);

void FORTRAN(pplldx)( int *, float *, float *, int *, 
                       char *, char *, float *, float * );


void FORTRAN(pplldc_envelope)(int *, float *, int *, int *, int *, int *,
                       int *, int *, float *, float *, int *, int *,
                       float *, float *, float *, float *, int *);

void FORTRAN(pplldc)( int *, float *, int *, int *, int *, int *, 
                       int *, int *, float *, float *, int *, int *, 
                       float *, float *, float *, float *, float *);

void FORTRAN(pplldv_envelope)(int *, float *, int *, int *, int *, 
                       int *, int *, int *);

void FORTRAN(pplldv)( int *, float *, int *, int *, int *, int *, 
                       int *, int *, float *);

void FORTRAN(save_ppl_memory_size)(int *);
void FORTRAN(get_ppl_memory_size)(int *);

void reallo_ppl_memory( int * );

