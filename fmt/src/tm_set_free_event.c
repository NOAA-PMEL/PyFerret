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

