/*
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
*/



/* this routine is called with either a 1 or a 0, and will set rl_event_hook
   accordingly.  rl_event_hook is used by readline and is the procedure it
   calls when waiting for input.  If this routine is called with a 0, 
   rl_event_hook is set to NULL, and readline will not process any event
   while it is waiting for input.  If the routine is called with a 1, 
   rl_event_hook is set to free_time.  readline will process free_time. */

/* had to add ifdef check for trailing underscore in routine name
   for aix port *kob* 10/94 */
/* 11/96 *kob* - Linux port - had to have double quotes around the STOP
                              message */

#ifdef unix

/* this routine will only work on a unix system */

#define NULL 0

typedef int Function ();
#ifdef NO_ENTRY_NAME_UNDERSCORES
void tm_set_free_event(n)
#else
void tm_set_free_event_(n)
#endif
int *n;
{
  extern Function *rl_event_hook;
#ifdef NO_ENTRY_NAME_UNDERSCORES
  void free_time();
#else
  void free_time_();
#endif



  if (*n) 
#ifdef NO_ENTRY_NAME_UNDERSCORES
    rl_event_hook = (Function *)free_time;
#else
    rl_event_hook = (Function *)free_time_;
#endif
  else
    rl_event_hook = (Function *)NULL;
  

}

#else
    STOP "TM_SET_FREE_EVENT isn't used by VMS"

#endif

