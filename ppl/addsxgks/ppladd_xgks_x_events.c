/* Routine to get waiting X events processed in xgks.  Called by
 * process_x_events, written when Solaris install failed to resize
 *
 * J Davison 3.8.94
 */

#ifdef NO_ENTRY_NAME_UNDERSCORES
xgks_x_events ()
#else
xgks_x_events_ ()
#endif
{
  xProcessEvents ();
}
