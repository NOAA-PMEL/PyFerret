/*
 * JC_InterInterface.h
 *
 * Jonathan Callahan
 * Jan 18'th 1996
 *
 * This file contains functions which allow interfaces in the Ferret GUI
 * to have effects on other interfaces.  It can be thought of as a central
 * hub where routing takes place.  It should be easier to maintain the
 * inter-interface connectivities if they are all contained in one file.
 *
 */


#ifndef _JC_INTERINTERFACE_H
#define _JC_INTERINTERFACE_H

/* .................... Includes .................... */

#include <Xm/Xm.h>
#include "UxXt.h"
 

/* .................... Defines .................... */

/*
extern Boolean JC_DefineVariable_is_displayed;
extern Boolean JC_SelectRegridding_is_displayed;
*/

/* .................... Function Declarations .................... */


extern int JC_II_Synchronize( swidget caller_id );

/*
 * Input     caller_id: the interface which is asking for a new dataset
 *    
 * Output    returns: [-2, -1, 0] for [fatal error, not found error, OK]
 */


extern void JC_II_SelectMenus_Recreate( swidget caller_id );

extern void JC_II_SynchronizeWindows( void );
extern void JC_II_FixRegridding( swidget caller_id );
extern void JC_II_ChangeRegriddingLabel( swidget caller_id );
extern void JC_II_MainMenu_Maintain( JC_StateFlags *SF_ptr );

#endif /* _JC_INTERINTERFACE_H */
