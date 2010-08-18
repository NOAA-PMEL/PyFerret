/**
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
 */
package gov.noaa.pmel.ferret.threddsBrowser;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Collection;
import java.util.Vector;

import javax.swing.JOptionPane;

import ucar.util.prefs.PreferencesExt;

/**
 * Maintains locations and filename extensions for a ThreddsBrowser.  Prompts the user for selecting
 * a location and, if applicable, filename extensions of datasets to be displayed in a ThreddsBrowser. 
 * @author Karl M. Smith - karl.smith (at) noaa.gov
 */
public class LocationSelector {

	/** the ThreddsBrowser associated with this location selector */
	private ThreddsBrowser tBrowser;

	/** list of saved locations */
	private Vector<String> savedLocations;

	/** Default directory for the local directory file chooser */ 
	private File localBrowseDir;

	/** A comma/semicolon/space separated list of acceptable filename extensions for local datasets */
	private String extensionsString;

	/**
	 * Maintains locations and filename extensions for the given ThreddsBrowser.  Prompts the user for selecting
	 * a location and, if applicable, filename extensions of datasets to be displayed in this ThreddsBrowser. 
	 */
	public LocationSelector(ThreddsBrowser tBrowser) {
		if ( tBrowser == null )
			throw new NullPointerException("null ThreddsBrowser given to the LocationSelector constructor");
		this.tBrowser = tBrowser;
		savedLocations = null;
		localBrowseDir = null;
		extensionsString = null;
	}

	/**
	 * Reset the default values in the browser to those given in the BrowserDefaults.
	 * @param defs the BrowserDefaults to use; cannot be null
	 */
	public void resetDefaults(BrowserDefaults defs) {
		// Set the saved locations
		Collection<String> locations = defs.getLocationStrings();
		if ( savedLocations == null ) {
			savedLocations = new Vector<String>(locations);
		}
		else {
			savedLocations.removeAllElements();
			savedLocations.addAll(locations);
		}

		// Get the directory for the local directory file chooser 
		localBrowseDir = defs.getLocalBrowseDir();

		// Get the string of acceptable filename extensions for displayed datasets
		extensionsString = defs.getExtensionsString();
	}

	/**
	 * Save all the current settings of this ThreddsBrowser to prefs.
	 */
	public void savePreferences(PreferencesExt prefs) {
		// Save the list of locations
		if ( savedLocations != null )
			BrowserDefaults.saveLocationsList(prefs, savedLocations);

		// Save the default directory for the local directory file chooser
		if ( localBrowseDir != null )
			BrowserDefaults.saveLocalBrowseDir(prefs, localBrowseDir);

		// Save the standard String of extensions
		if ( extensionsString != null )
			BrowserDefaults.saveExtensionsString(prefs, extensionsString);
	}

	/**
	 * Request a new location to be displayed in the associated ThreddsBrowser.
	 * If args is not null and not empty, uses args[0] as the new location to be shown;
	 * otherwise, brings up a dialog prompting the user to select a new location.
	 * The location can be a URI string, URL string, or local directory pathname.
	 * If the location is a local directory, the user will then be prompted for the
	 * filename extensions of datasets to display.
	 */
	public void selectLocation(String [] args) {
		// Get the location to be shown from the arguments, if given
		String location = null;
		if ( (args != null) && (args.length > 0) ) {
			location = args[0].trim();
			if ( location.isEmpty() )
				location = null;
		}

		// If arguments not given or the first argument is blank, prompt the user for the location
		if ( location == null ) {
			LocationSelectorDialog selectorDialog = new LocationSelectorDialog(tBrowser, savedLocations, localBrowseDir);
			location = selectorDialog.selectLocation();
			selectorDialog.dispose();
		}
		// Just return if user canceled out of the dialog
		if ( location == null ) 
			return;

		// Add/move this location to the top of the saved list.
		if ( savedLocations == null )
			savedLocations = new Vector<String>();
		else
			savedLocations.remove(location);
		savedLocations.add(0, location);

		// Create a URI from the given location string
		URI uri;
		try {
			uri = new URI(location);
		} catch (URISyntaxException e) {
			// If problems, assume this is a local file pathname
			File locFile = new File(location);
			uri = locFile.toURI();
		}

		// Check if local directory or a Thredds server
		String scheme = uri.getScheme();
		if ( (scheme == null) || "file".equals(scheme) ) {
			// Save this new default location for the local file browser
			localBrowseDir = new File(uri.getPath());
			// Get the filename extensions of datasets in this local directory
			String newExtensions = JOptionPane.showInputDialog(tBrowser, "Show datasets with the filename extension\n" +
															   "(give none to show all files):", extensionsString);
			// Just return if user canceled out of the dialog
			if ( newExtensions == null )
				return;
			// Clean and save the new extensions
			Collection<String> extsColl = BrowserDefaults.parseExtensionsString(newExtensions);
			extensionsString = BrowserDefaults.createExtensionsString(extsColl);
			// Show the datasets in this local directory
			tBrowser.showLocalLocation(localBrowseDir, new ExtensionFileFilter(extsColl));
		}
		else {
			// Show the datasets from this Thredds server 
			tBrowser.showThreddsServerLocation(location);
		}
	}

}
