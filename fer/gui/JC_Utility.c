/*
 * JC_Utility.c
 *
 * Jonathan Callahan
 * Dec 19th 1995
 *
 * This file contains utility functions for common tasks.
 *
 * I have endeavored to stay very close to true Object-Oriented-Programming
 * principles and have in general followed the C++ guidelines as I learned
 * them from "Weiskamp & Flamig:  The complete C++ primer".
 *
 * 96.12.12 Removed #ifdef FULL_GUI_VERSION which used my_secs_to_date in 
 *          JC_String_CreateTimeLabel when not set.
 *
 */


/* .................... Includes .................... */

#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/RowColumn.h>
#include <Xm/CascadeB.h>
#include <Xm/CascadeBG.h>
#include <Xm/SeparatoG.h>
#include <Xm/ToggleB.h>
#include <Xm/ToggleBG.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>

#include "UxXt.h" /* for UxTopLevel used in the SGI_POPUS section */

#include "ferret_structures.h"
#include "ferret_fortran.h"
#include "ferret_shared_buffer.h"


/* .................... Defines .................... */

#define MAX_MENU_ITEMS 64


extern JC_StateFlags GLOBAL_StateFlags;
extern LIST *GLOBAL_DatasetList;


#ifdef SGI_POPUPS
Window windows[10];
int window_count=0;
#endif /* SGI_POPUPS */

/* .................... Typedefs .................... */

typedef struct _JC_menu_item {
  char *label;		/* the label for the item */
  WidgetClass *class;	/* pushbutton, label, separator... */
  char mnemonic;	/* mnemonic; NULL if none */
  char *accelerator;	/* accelerator; NULL if none */
  char *accel_text;	/* to be converted to compound string */
  void (*callback)();	/* routine to call; NULL if none */
  char *callback_data;	/* client_data for callback() */
  struct _JC_menu_item *subitems; /* pullright menu items, if not NULL */
} JC_MenuItem;


/* .................... External Declarations .................... */

extern void TimeToFancyDate(double *val, char *outDate);
extern char *CollectToReturn(char *targetStr, char *subStr);
extern int  ferret_query(int query, smPtr sBuffer, char *tag,
		 char *arg1, char *arg2, char *arg3, char *arg4 );


/* .................... Internal Declarations .................... */

static void JC_String_RemoveBlanks( char *string );
       void JC_String_RemoveWhiteSpace( char *string );

static void JC_String_CreateLongitudeLabel( char *string, double value );
static void JC_String_CreateLatitudeLabel( char *string, double value );
static void JC_String_CreateTimeLabel( char *string, double value );

static float JC_String_ConvertLongitudeToFloat( char *string );
static float JC_String_ConvertLatitudeToFloat( char *string );
static float JC_String_ConvertTimeToFloat( char *string, JC_Span *S_ptr );

static float JC_DateString_ConvertToSeconds( char *string, JC_Span *S_ptr );
static int JC_DateString_MonthToInt( char *string );

static char separator_string[] = "separator";
static JC_MenuItem subitems[MAX_MENU_ITEMS][MAX_MENU_ITEMS]={ NULL, };


/* .................... JC_Menu methods .................... */

/*
 * This is the BuildMenu function from p. 559 of The O'Reilly X Series, vol. Six A.
 *
 * It has been copied verbatim; only the name has been changed.
 */

/* Build popup, option and pulldown menus, depending on the menu_type.
 * It may be XmMENU_PULLDOWN, XmMENU_OPTION or XmMENU_POPUP.  Pulldowns
 * return the CascadeButton that pops up the menu.  Popups return the menu.
 * Option menus are created, but the RowColumn that acts as the option
 * "area" is returned unmanaged. (The user must manage it.)
 * Pulldown menus are built from cascade buttons, so this function
 * also builds pullright menus.  The function also adds the right
 * callback for PushButton or ToggleButton menu items.
 */
Widget
JC_Menu_Build(parent, menu_type, menu_title, menu_mnemonic, tear_off, items )
Widget parent;
int menu_type;
char *menu_title, menu_mnemonic;
Boolean tear_off;
JC_MenuItem *items;
{
     Widget menu, cascade, widget;
     int i;
     XmString str;
     ArgList args;

#ifdef SGI_POPUPS
     Arg al[20]; /* arg list */
     int ac; /* arg count */
#endif /* SGI_POPUPS */


     if (menu_type == XmMENU_PULLDOWN || menu_type == XmMENU_OPTION) {
#ifdef SGI_POPUPS
       SG_getPopupArgs (XtDisplay(parent), NULL, al, &ac);
       menu = XmCreatePulldownMenu (parent, "_pulldown", al, ac);
       windows[window_count++] = XtWindow(UxTopLevel);
#else /* normal way */
       menu = XmCreatePulldownMenu (parent, "_pulldown", NULL, 0);
#endif /*SGI_POPUPS */
     } else if (menu_type == XmMENU_POPUP)
	  menu = XmCreatePopupMenu (parent, "_popup", NULL, 0);
     else {
	  XtWarning ("Invalid menu type passed to JC_Menu_Build()");
	  return NULL;
     }
     if (tear_off)
	  XtVaSetValues (menu, XmNtearOffModel, XmTEAR_OFF_ENABLED, NULL);

     /* Pulldown menus require a cascade button to be made */
     if (menu_type == XmMENU_PULLDOWN) {
	  str = XmStringCreateLocalized (menu_title);
	  cascade = XtVaCreateManagedWidget (menu_title,
					     xmCascadeButtonGadgetClass, parent,
					     XmNsubMenuId, menu,
					     XmNlabelString, str,
					     XmNmnemonic,menu_mnemonic,
					     NULL);
	  XmStringFree (str);
     }
     else if (menu_type == XmMENU_OPTION) {
	  /* Option menus are a special case, but not hard to handle */
	  Arg args[5];
	  int n = 0;
	  str = XmStringCreateLocalized (menu_title);
	  XtSetArg (args[n], XmNsubMenuId, menu); n++;
	  XtSetArg (args[n], XmNlabelString, str); n++;
	  /* This isn't a cascade, but this is the widget handle
	   * we're going to return at the end of the function.
	   */
	  cascade = XmCreateOptionMenu (parent, menu_title, args, n);
	  XmStringFree (str);
     }

     /* Now add the menu_items */
     for (i = 0; items[i].label != NULL; i++) {
       /* If subitems exist, create the pull-right menu by calling this
	* function recursively.  Since the function returns a cascade
	* button, the widget returned is used.
	*/
       if ( i == MAX_MENU_ITEMS ) { /* check added by JC */
	 fprintf(stderr, "ERROR in JC_Utility: JC_Menu_Build(): %d items == MAX_MENU_ITEMS\n", i);
	 continue;
       }

       if (items[i].subitems)
	 if (menu_type == XmMENU_OPTION) {
	   XtWarning ("You can't have submenus from option menu items.");
	   continue;
	 }
	 else
	   widget = JC_Menu_Build (menu, XmMENU_PULLDOWN, items[i].label,
				   items[i].mnemonic, tear_off, items[i].subitems);
       else
 	 widget = XtVaCreateManagedWidget (items[i].label, 
 					   *items[i].class, menu, 
 					   NULL);        
     
       /* Whether the item is a real item or a cascade button with a
        * menu, it can still have a mnemonic.
        */
       if (items[i].mnemonic)
	 XtVaSetValues (widget, XmNmnemonic, items[i].mnemonic, NULL);
	  
       /* any item can have an accelerator, except cascade menus. But,
        * we don't worry about that; we know better in our declarations.
        */
       if (items[i].accelerator) {
	 str = XmStringCreateLocalized (items[i].accel_text);
	 XtVaSetValues (widget,
			XmNaccelerator, items[i].accelerator,
			XmNacceleratorText, str,
			NULL);
	 XmStringFree (str);
       }
	  
       if (items[i].callback) {
	 XtAddCallback (widget,
			(items[i].class == &xmToggleButtonWidgetClass ||
			 items[i].class == &xmToggleButtonGadgetClass) ?
			XmNvalueChangedCallback : /* ToggleButton class */
			XmNactivateCallback, /* PushButton class */
			items[i].callback, items[i].callback_data);
       }
     }
     
     /* for popup menus, just return the menu; pulldown menus, return
      * the cascade button; option menus, return the thing returned
      * from XmCreateOptionMenu().  This isn't a menu, or a cascade button!
      */
     return menu_type == XmMENU_POPUP ? menu : cascade;
}


void JC_Menu_AddDsetVars( JC_MenuItem *items, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)() )
{


     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     int numDataSets=list_size(GLOBAL_DatasetList);
     int i=0, j=0, k=0;
     JC_DatasetElement *DE_ptr;

/*
 * - For each dataset:
 *      get the variables
 *      for each variable
 *         create the "variable" pushButton item with: label, class, callback, callback_data
 *
 *   create the "dataset" cascadeButton with: label, class, subitems
 */


     for ( i=0, list_mvfront(GLOBAL_DatasetList); i<numDataSets; i++, list_mvnext(GLOBAL_DatasetList)  ) {

	  DE_ptr = (JC_DatasetElement *)list_curr(GLOBAL_DatasetList);

	  k = 0;

	  /*
	   * Add file variables here
	   */

	  for (j=0, list_mvfront(DE_ptr->varList); j<list_size(DE_ptr->varList); j++, list_mvnext(DE_ptr->varList), k++) {

	       subitems[i][k].label = list_curr(DE_ptr->varList);
	       subitems[i][k].class = &xmPushButtonWidgetClass;
	       subitems[i][k].callback = var_fn_ptr;
	       subitems[i][k].callback_data = (XtPointer) ((JC_DatasetElement *)list_curr(GLOBAL_DatasetList))->name;
	       
	  }
	 
	  subitems[i][k].label = separator_string;
	  subitems[i][k].class = &xmSeparatorGadgetClass;
	  subitems[i][k].callback = NULL;
	  subitems[i][k].callback_data = (XtPointer)NULL;
	  k++;
	
	  /*
	   * Add cloned variables here
	   */

	  for (j=0, list_mvfront(DE_ptr->cvarList); j<list_size(DE_ptr->cvarList); j++, list_mvnext(DE_ptr->cvarList), k++) {

	       subitems[i][k].label = ( (JC_Object *)list_curr(DE_ptr->cvarList) )->name;
	       subitems[i][k].class = &xmPushButtonWidgetClass;
	       subitems[i][k].callback = cvar_fn_ptr;
	       subitems[i][k].callback_data = (XtPointer) ((JC_DatasetElement *)list_curr(GLOBAL_DatasetList))->name;
	       
	  }
	 
	  subitems[i][k].label = separator_string;
	  subitems[i][k].class = &xmSeparatorGadgetClass;
	  subitems[i][k].callback = NULL;
	  subitems[i][k].callback_data = (XtPointer)NULL;
	  k++;
	 
	  /*
	   * Add user-defined variables here
	   */

	  for (j=0, list_mvfront(DE_ptr->dvarList); j<list_size(DE_ptr->dvarList); j++, list_mvnext(DE_ptr->dvarList), k++) {

	       subitems[i][k].label = ( (JC_DefinedVariable *)list_curr(DE_ptr->dvarList) )->name;
	       subitems[i][k].class = &xmPushButtonWidgetClass;
	       subitems[i][k].callback = dvar_fn_ptr;
	       subitems[i][k].callback_data = (XtPointer) ((JC_DatasetElement *)list_curr(GLOBAL_DatasetList))->name;
	       
	  }

	  subitems[i][k].label = NULL;
	  
	  items[i].label = ((JC_DatasetElement *)list_curr(GLOBAL_DatasetList))->name;
	  items[i].class = &xmCascadeButtonWidgetClass;
	  items[i].subitems = (JC_MenuItem *)subitems[i];
	  
     }
     
     items[i].label = NULL;
     items[i].class = NULL;
     items[i].callback = NULL;
     items[i].callback_data = (XtPointer)NULL;
     
}


void JC_Menu_AddDsetUvars( JC_MenuItem *items, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)() )
{
     JC_StateFlags *SF_ptr=&GLOBAL_StateFlags;
     int numDataSets=list_size(GLOBAL_DatasetList);
     int i=0, j=0, k=0;
     JC_DatasetElement *DE_ptr;

/*
 * - For each dataset:
 *      get the user defined variables
 *      for each variable
 *         create the "variable" pushButton item with: label, class, callback, callback_data
 *
 *   create the "dataset" cascadeButton with: label, class, subitems
 */


     for ( i=0, list_mvfront(GLOBAL_DatasetList); i<numDataSets; i++, list_mvnext(GLOBAL_DatasetList)  ) {
	  
	  DE_ptr = (JC_DatasetElement *)list_curr(GLOBAL_DatasetList);

	  k = 0;

	  /*
	   * Add user-defined variables here
	   */

	  for (j=0, list_mvfront(DE_ptr->dvarList); j<list_size(DE_ptr->dvarList); j++, list_mvnext(DE_ptr->dvarList), k++) {

	       subitems[i][k].label = ( (JC_DefinedVariable *)list_curr(DE_ptr->dvarList) )->name;
	       subitems[i][k].class = &xmPushButtonWidgetClass;
	       subitems[i][k].callback = dvar_fn_ptr;
	       subitems[i][k].callback_data = (XtPointer) ((JC_DatasetElement *)list_curr(GLOBAL_DatasetList))->name;
	       
	  }

	  subitems[i][k].label = NULL;
	  
	  items[i].label = ((JC_DatasetElement *)list_curr(GLOBAL_DatasetList))->name;
	  items[i].class = &xmCascadeButtonWidgetClass;
	  items[i].subitems = (JC_MenuItem *)subitems[i];
	  
     }

     items[i].label = NULL;
     
     
}


/* .................... JC_String methods .................... */


static void JC_String_RemoveBlanks( char *string )
{
     char tempText[MAX_NAME_LENGTH]="";
     char *s=string, *t=tempText;

/*
 * - While the string has not ended
 *      If the character is a blank
 *         point to the next character
 *      Else
 *         copy the char to tempText and increment both
 *
 *   Copy tempText back to string
 */

     while ( *s != '\0' ) {
	  if ( *s == ' ' )
	       s++;
	  else
	       *t++ = *s++;
     }
     *t = '\0';

     strcpy(string, tempText);
}


void JC_String_RemoveWhiteSpace( char *string )
{
     char tempText[MAX_NAME_LENGTH]="";
     char *s=string, *t=tempText;

/*
 * - While the string has not ended
 *      If the character is a blank, tab or newline
 *         point to the next character
 *      Else
 *         copy the char to tempText and increment both
 *
 *   Copy tempText back to string
 */

     while ( *s != '\0' ) {
	  if ( *s == ' '  || *s == '\t' || *s == '\n' )
	       s++;
	  else
	       *t++ = *s++;
     }
     *t = '\0';

     strcpy(string, tempText);
}


int JC_String_EndsWithTag( char *string, char *tag )
{
     int i=0, j=0, full_length=0, tag_length=0;

     full_length = strlen(string);
     tag_length = strlen(tag);

     if ( full_length <= tag_length ) return FALSE;
	
     for (i=full_length-tag_length, j=0; i<full_length; i++, j++)
	  if (string[i] != tag[j] ) return FALSE;

     return TRUE;
}


void JC_String_CreateFancyFerretLabel( char *string, double value, JC_Span *S_ptr, JC_StateFlags *SF_ptr )
{
     char tempText[MAX_NAME_LENGTH]="";

     if ( S_ptr->xyzt == X_AXIS ) {

	  JC_String_CreateLongitudeLabel(string, value);
	  JC_String_RemoveBlanks(string);

     } else if ( S_ptr->xyzt == Y_AXIS ) {
	  
	  JC_String_CreateLatitudeLabel(string, value);
	  JC_String_RemoveBlanks(string);
	  
     } else if ( S_ptr->xyzt == Z_AXIS ) {
	  
	  sprintf(string, "%.2f", value);
	  
     } else if ( S_ptr->xyzt == T_AXIS ) {
	  
	  if ( S_ptr->time_type == MODEL_TIME )
	       sprintf(string, "%.2f", value);
	  else {
	       value = S_ptr->secsAtT0 + S_ptr->secsPerUnit * value;
	       TimeToFancyDate(&value, tempText); /*JC_String_CreateTimeLabel(&value, tempText);*/
	       if ( SF_ptr->time_resolution_includes_hours )
		    sprintf(string, "\"%s\"", tempText);
	       else
		    sprintf(string, "\"%s\"", strtok(tempText, ":"));
	  }
	  
     }
     
}


void JC_String_CreateFancyLabel( char *string, double value, JC_Span *S_ptr, JC_StateFlags *SF_ptr )
{
     char tempText[MAX_NAME_LENGTH]="";

     if ( S_ptr->xyzt == X_AXIS )

	  JC_String_CreateLongitudeLabel(string, value);

     else if ( S_ptr->xyzt == Y_AXIS )

	  JC_String_CreateLatitudeLabel(string, value);

     else if ( S_ptr->xyzt == Z_AXIS )

	  sprintf(string, "%.2f", value);

     else if ( S_ptr->xyzt == T_AXIS ) {

	  if ( S_ptr->time_type == MODEL_TIME )
	       sprintf(string, "%.2f", value);
	  else {
	       value = S_ptr->secsAtT0 + S_ptr->secsPerUnit * value;
	       TimeToFancyDate(&value, tempText); /*JC_String_CreateTimeLabel(&value, tempText);*/
	       if ( SF_ptr->time_resolution_includes_hours )
		    strcpy(string, tempText);       
	       else
		    strcpy(string, strtok(tempText, ":"));
	  }

     }

}


static void JC_String_CreateLongitudeLabel( char *string, double value )
{

  while ( value < 0 ) {
    value += 360.0;
  }

  if ( value <= 180.0 )
    sprintf(string, "%.2f E", value);
  else if ( value > 180.0 && value <= 360.0 )
    sprintf(string, "%.2f W", 360.0 - value);
  else if ( value > 360.0 && value <= 720.0)
    sprintf(string, "%.2f E", value);
  else
    sprintf(string, "%.2f", value);
     
     
}

static void JC_String_CreateLatitudeLabel( char *string, double value )
{

     if ( value < 0.0 )
	  sprintf(string, "%.2f S", -value);
     else
	  sprintf(string, "%.2f N", value);

}


static void JC_String_CreateTimeLabel( char *string, double value )
{

#ifdef NO_ENTRY_NAME_UNDERSCORES
  secs_to_date_c(value, string);
#else
  secs_to_date_c_(value, string);
#endif

  FixDate(string);

}


static float JC_String_ConvertLongitudeToFloat( char *string )
{
     float value=0;
     int n=0;
     char tempText[MAX_NAME_LENGTH]="", hemis='\0';

     if ( (n = sscanf(string, "%f%s", &value, tempText)) != 2 ) {
	  fprintf(stderr, "ERROR in JC_Utility.c: JC_String_ConvertLongitudeToFloat(): sscanf read %d items.\n", n);
	  return (float)INTERNAL_ERROR;
     }

     if ( strchr(tempText, 'E') )
	  hemis = 'E';
     else
	  hemis = (char) toupper(tempText[0]);

     if (value < -180.0 || value > 720.0)
	  return (float)INTERNAL_ERROR;
     
     if (value <= 0)
	  value += 360;
     
     switch (hemis) {
	case '\0':
	case 'E':
	  return value;
	  break;
	case 'W':
	  return (360 - value);
	  break;
	default:
	  return (float)INTERNAL_ERROR;
	  break;
     }

}

static float JC_String_ConvertLatitudeToFloat( char *string )
{
     float value=0;
     int n=0;
     char tempText[MAX_NAME_LENGTH]="", hemis='\0';

     if ( (n = sscanf(string, "%f%s", &value, tempText)) != 2 ) {
	  fprintf(stderr, "ERROR in JC_Utility.c: JC_String_ConvertLatitudeToFloat(): sscanf read %d items.\n", n);
	  return (float)INTERNAL_ERROR;
     }

     hemis = (char) toupper(tempText[0]);
    
     if (value < -90.0 || value > 90.0) 
	  return (float)INTERNAL_ERROR;

     if (value <= 0)
	  return value;

     switch (hemis) {
	case '\0':
	case 'N':
	  return value;
	  break;
	case 'S':
	  return -value;
	  break;
	default:
	  return (float) INTERNAL_ERROR;
	  break;
     }

}


static float JC_String_ConvertTimeToFloat( char *string, JC_Span *S_ptr )
{
     double value=0.0;
     int n=0;
     char tempText[MAX_NAME_LENGTH]="";

     if ( S_ptr->time_type == MODEL_TIME ) {

	  if ( (n = sscanf(string, "%f%s", &value, tempText)) != 1 ) {
	       fprintf(stderr, "ERROR in JC_Utility.c: JC_String_ConvertTimeToFloat(): sscanf read %d items.\n", n);
	       value = (double)INTERNAL_ERROR;
	  }

     } else if ( S_ptr->time_type == CLIMATOLOGY_TIME ) {
	  
	  value = (double)JC_DateString_ConvertToSeconds(string, S_ptr);
	  value = value/S_ptr->secsPerUnit - S_ptr->secsAtT0;
	  
     } else if ( S_ptr->time_type == CALENDAR_TIME ) {
	  
	  value = (double)JC_DateString_ConvertToSeconds(string, S_ptr);
	  value = value/S_ptr->secsPerUnit - S_ptr->secsAtT0/S_ptr->secsPerUnit;
	  
     } else {
	  fprintf(stderr, "ERROR in JC_Utility.c: String_ConvertTimeToFloat(): time_type = %d\n", S_ptr->time_type);
     }

     return (float)value;
}


float JC_String_ConvertToFloat( char *string, JC_Span *S_ptr )
{
     float value=0;
     int   n=0;
     char  tempText[MAX_NAME_LENGTH]="";
     
     if (strcmp(string, "") == 0)
	  return (float)INTERNAL_ERROR;
     
     switch ( S_ptr->xyzt ) {
	  
	case X_AXIS:
	  value = JC_String_ConvertLongitudeToFloat(string);
	  return value;
	  break;
	  
	case Y_AXIS:
	  value = JC_String_ConvertLatitudeToFloat(string);
	  return value;
	  break;

	case Z_AXIS:
	  if ( (n = sscanf(string, "%f%s", &value, tempText)) != 1 ) {
	       fprintf(stderr, "ERROR in JC_Utility.c: JC_String_ConvertToFloat(): sscanf read %d items.\n", n);
	       return (float)INTERNAL_ERROR;
	  }
	  if (value < 0.0) 
	       return (float)INTERNAL_ERROR;
	  else
	       return value;
	  break;

	case T_AXIS:
	  value = JC_String_ConvertTimeToFloat(string, S_ptr);
	  return value;
	  break;
	  
     }
     
}


/* .................... Dealing with date strings .................... */


static float JC_DateString_ConvertToSeconds( char *string, JC_Span *S_ptr )
{
     char tempText[MAX_NAME_LENGTH]="", *substring;
     char separators[]="-:/ ";
     int day=0, month=0, year=0, hour=0, minute=0, second=0;
     float seconds_since_year_zero=0;

/*
 * - Copy the string locally because strtok(s1,s2) alters s1.
 */
     strcpy(tempText, string);

/*
 * - Get the first substring.
 */
     if ( (substring = strtok(tempText, separators)) == NULL )
	  goto done_parsing;
     
/*
 * - If the first substring is a number:
 *    - get the days
 *    - get the month
 *
 * - Else:
 *    - get the month
 *    - get the days
 */

     if ( isdigit(substring[0]) ) {

	  if ( sscanf(substring, "%d", &day) > 1 ) {
	       fprintf(stderr, "ERROR in JC_Utility.c: DateString_ConvertToSeconds(): cannot scan \"%s\"\n", substring);
	       return INTERNAL_ERROR;
	  }
     
	  if ( (substring = strtok(NULL, separators)) == NULL )
	       goto done_parsing;
     
	  if ( (month = JC_DateString_MonthToInt(substring)) == INTERNAL_ERROR )
	       return INTERNAL_ERROR;
     
     } else {
	  
	  if ( (month = JC_DateString_MonthToInt(substring)) == INTERNAL_ERROR )
	       return INTERNAL_ERROR;
	  
	  if ( (substring = strtok(NULL, separators)) == NULL )
	       goto done_parsing;
	  
	  if ( sscanf(substring, "%d", &day) > 1 ) {
	       fprintf(stderr, "ERROR in JC_Utility.c: DateString_ConvertToSeconds(): cannot scan \"%s\"\n", substring);
	       return INTERNAL_ERROR;
	  }

     }

/*
 * - If we have calendar_time, get the year.
 */
     if ( S_ptr->time_type == CALENDAR_TIME ) {
	  if ( (substring = strtok(NULL, separators)) == NULL )
	       goto done_parsing;

	  if ( sscanf(substring, "%d", &year) > 1 ) {
	       fprintf(stderr, "ERROR in JC_Utility.c: DateString_ConvertToSeconds(): cannot scan \"%s\"\n", substring);
	       return INTERNAL_ERROR;
	  }
     }
     
/*
 * - Get the hour.
 */
     if ( (substring = strtok(NULL, separators)) == NULL )
	  goto done_parsing;
     
     if ( sscanf(substring, "%d", &hour) > 1 ) {
	  fprintf(stderr, "ERROR in JC_Utility.c: DateString_ConvertToSeconds(): cannot scan \"%s\"\n", substring);
	  return INTERNAL_ERROR;
     }
     
/*
 * - Get the minute.
 */
     if ( (substring = strtok(NULL, separators)) == NULL )
	  goto done_parsing;
     
     if ( sscanf(substring, "%d", &minute) > 1 ) {
	  fprintf(stderr, "ERROR in JC_Utility.c: DateString_ConvertToSeconds(): cannot scan \"%s\"\n", substring);
	  return INTERNAL_ERROR;
     }
     
/*
 * - Get the second.
 */
     if ( (substring = strtok(NULL, separators)) == NULL )
	  goto done_parsing;
     
     if ( sscanf(substring, "%d", &second) > 1 ) {
	  fprintf(stderr, "ERROR in JC_Utility.c: DateString_ConvertToSeconds(): cannot scan \"%s\"\n", substring);
	  return INTERNAL_ERROR;
     }
     
     
   done_parsing:
     
#ifdef NO_ENTRY_NAME_UNDERSCORES
     seconds_since_year_zero = (float)tm_secs_from_bc(&year, &month, &day, &hour, &minute, &second);
#else
     seconds_since_year_zero = (float)tm_secs_from_bc_(&year, &month, &day, &hour, &minute, &second);
#endif
     
     return seconds_since_year_zero;
     
     
}


static int JC_DateString_MonthToInt( char *string )
{
     char tempText[MAX_NAME_LENGTH]="";
     int i=0;

     strcpy(tempText, string);

     for ( i=0; i<strlen(tempText); i++ )
	  tempText[i] = (char)toupper(tempText[i]);

     if (strstr(tempText, "JAN")) return 1;
     else if (strstr(tempText, "FEB")) return 2;
     else if (strstr(tempText, "MAR")) return 3;
     else if (strstr(tempText, "APR")) return 4;
     else if (strstr(tempText, "MAY")) return 5;
     else if (strstr(tempText, "JUN")) return 6;
     else if (strstr(tempText, "JUL")) return 7;
     else if (strstr(tempText, "AUG")) return 8;
     else if (strstr(tempText, "SEP")) return 9;
     else if (strstr(tempText, "OCT")) return 10;
     else if (strstr(tempText, "NOV")) return 11;
     else if (strstr(tempText, "DEC")) return 12;
     else {
	  fprintf(stderr, "ERROR: JC_Utility.c: DateString_MonthToInt(): \"%s\" not recognized as month.\n", tempText);
	  return INTERNAL_ERROR;
     }

}







