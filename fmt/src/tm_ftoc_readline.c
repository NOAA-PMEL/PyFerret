/* tm_ftoc_readline -- based on "manexamp.c" in the readline distribution. */
/* c jacket  to make gnu readline callable from FORTRAN */

/* had to add ifdef check for trailing underscore in routine name
   for aix port *kob* 10/94 */

#include <stdio.h>
#include "readline/readline.h"

/* A static variable for holding the line. */
static char *line_read = (char *)NULL;

/* Read a string, and return a pointer to it.  Returns NULL on EOF. */
char *do_gets ( prompt )
  char *prompt;

{
  /* If the buffer has already been allocated, return the memory
     to the free pool. */
  if (line_read != (char *)NULL)
    {
      free (line_read);
      line_read = (char *)NULL;
    }

  /* Get a line from the user. */
  line_read = readline (prompt);

  /* If the line has any text in it, save it on the history. */
  if (line_read && *line_read)
    add_history (line_read);

  return (line_read);
}

#ifdef NO_ENTRY_NAME_UNDERSCORES
tm_ftoc_readline( prompt, buff )
#else
tm_ftoc_readline_( prompt, buff )
#endif
/* c jacket routine to make gnu readline callable from FORTRAN */
  char *prompt, *buff;
{
  char *ptr;

/* invoke gnu readline with line recall and editing */
  ptr = do_gets ( prompt );

/* copy the line into the buffer provided from FORTRAN */
  if (ptr != (char *)NULL)
    strcpy( buff, ptr );
  else
    buff[0] = '\004';   /* ^D  */

  return (0);
}
