/* JC_Utility.c
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
 */


#ifndef _JC_UTILITY_H
#define _JC_UTILITY_H

/* .................... Includes .................... */

#include <Xm/Xm.h>
#include <Xm/RowColumn.h>
#include <Xm/CascadeBG.h>
#include <Xm/ToggleB.h>
#include <Xm/ToggleBG.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>

#include "ferret_structures.h"


/* .................... Defines .................... */

#define MAX_MENU_ITEMS 64


/* .................... Variables .................... */

#ifdef SGI_POPUPS
extern Window windows[10];
extern int window_count;
#endif /* SGI_POPUPS */

/* .................... Typedefs .................... */

typedef struct _JC_menu_item {
     char *label;		/* the label for the item */
     WidgetClass *class;	/* pushbutton, label, separator... */
     char mnemonic;		/* mnemonic; NULL if none */
     char *accelerator;		/* accelerator; NULL if none */
     char *accel_text;		/* to be converted to compound string */
     void (*callback)();	/* routine to call; NULL if none */
     char *callback_data;	/* client_data for callback() */
     struct _JC_menu_item *subitems; /* pullright menu items, if not NULL */
} JC_MenuItem;


/* .................... JC_Menu methods .................... */


extern Widget JC_Menu_Build( Widget parent, int menu_type, char *menu_title, char *menu_mnemonic, Boolean tear_off, JC_MenuItem *items );

/*
 * Input     parent:         parent widget
 *           menu_type:      one of: XmMENU_PULLDOWN, XmMENU_OPTION or XmMENU_POPUP
 *           menu_title:     title for the menu
 *           menu_mnemonic:  mnemonic for the menu
 *           tear_off:       is this a tear_off menu?
 *           items:          JC_MenuItem structure containing menu items
 *    
 * Output    Builds popup, option and pulldown menus, depending on the menu_type.
 *           It may be XmMENU_PULLDOWN, XmMENU_OPTION or XmMENU_POPUP.  Pulldowns
 *           return the CascadeButton that pops up the menu.  Popups return the menu.
 *           Option menus are created, but the RowColumn that acts as the option
 *           "area" is returned unmanaged. (The user must manage it.)
 *           Pulldown menus are built from cascade buttons, so this function
 *           also builds pullright menus.  The function also adds the right
 *           callback for PushButton or ToggleButton menu items.
 */

/*
 * Input     items:    JC_MenuItem structure 
 *           subitems: JC_MenuItem structure
 *    
 * Output    Frees all the callback_data which which menu items point to.
 *           The callback_data must have been allocated with XtNewString or another Xt memory allocator.
 */


extern void JC_Menu_AddDsetVars( JC_MenuItem *items, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)() );
extern void JC_Menu_AddDsetUvars( JC_MenuItem *items, void (*var_fn_ptr)(), void (*dvar_fn_ptr)(), void (*cvar_fn_ptr)() );



/* .................... JC_String methods .................... */


extern void JC_String_CreateFancyFerretLabel( char *string, double value, JC_Span *S_ptr, JC_StateFlags *SF_ptr );

/*
 * Input     string:  string which will have a fancy label inserted
 *           value:   value which is to be converted to a string
 *           S_ptr:   pointer to the span for which this value is being converted
 *           SF_ptr:  pointer to the current State Flags
 *    
 * Output    Creates a fancy Ferret label for the current axis(span) equivalent to 'value'.
 *           All blanks are removed from the string.
 */

extern void JC_String_CreateFancyLabel( char *string, double value, JC_Span *S_ptr, JC_StateFlags *SF_ptr );

/*
 * Input     string:  string which will have a fancy label inserted
 *           value:   value which is to be converted to a string
 *           S_ptr:   pointer to the span for which this value is being converted
 *           SF_ptr:  pointer to the current State Flags
 *    
 * Output    Creates a fancy label for the current axis(span) equivalent to 'value'.
 *           This label has a space between the value and the hemisphere.
 */

extern float JC_String_ConvertToFloat( char *string, JC_Span *S_ptr );

/*
 * Input     string:  string which will be interpreted as a "ww" value
 *           A_ptr:   pointer to the JC_Axis with which the string is associated
 *    
 * Output    Returns a "ww" value corresponding to the text string received.
 */

extern void JC_String_RemoveWhiteSpace( char *string );

/*
 * Input     string:  string which will have all blanks, tabs and newlines removed
 *    
 * Output     
 */

extern int JC_String_EndsWithTag( char *string, char *tag );

/*
 * Input     string:
 *           tag:     tag to be found in 'string'
 *    
 * Output    Returns TRUE if 'string' ends with 'tag', FALSE otherwise
 */



#endif /* _JC_UTILITY_H */

/* ~~~~~~~~~~~~~~~~~~~~ END OF JC_CommandGen.h ~~~~~~~~~~~~~~~~~~~~ */
