/* Gigal: An open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
**
** This program is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program. If not, see <http://www.gnu.org/licenses/>.
**
** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
*/

#include <string>

#include "instanceStream.h"

/**
<!-- globalinfo-start -->
 * A class that describes the capabilites (e.g., handling certain types of
 * attributes, missing values, types of classes, etc.) of a specific
 * learner. By default, the learner is capable of nothing.<br/>
 <!-- globalinfo-end -->
 * 
 * @author Ana M. Martinez (anam.martinez@monash.edu)
 */

class capabilities {
public:
  capabilities();
  virtual ~capabilities();
  
  /**
   * Prints an error message if the method (this object) can not handle the data in the 
   * InstanceStream is. 
   * 
   * @param is training data to process
   * @param methodName the name of the method (e.g. learner, filter, ...)
   */
  void testCapabilities(InstanceStream &is, const std::string& methodName); 
  
  void setCatAtts(bool catAtts);                        ///< set whether the method can handle categorical attributes  
  void setNumAtts(bool numAtts);                        ///< set whether the method can handle numeric attributes    
//  void setmissingValues(bool missingValues);            ///< set whether the method can handle missing values      
//  void setNoClass(bool noClass);                        ///< set whether the method can handle data without a given class (e.g. clustering)        
//  void setCatClass(bool catClass);                      ///< set whether the method can handle categorical classes 
//  void setBinaryClassOnly(bool binaryClassOnly);        ///< set whether the method can handle binary classes  
//  void setNumClass(bool numClass);                      ///< set whether the method can handle numeric classes  
//  void setMissingClassValues_(bool missingClassValues); ///< set whether the method can handle missing values for the class
  
private:
  bool catAtts_;            ///< true if the method can handle categorical attributes  
  bool numAtts_;            ///< true if the method can handle numeric attributes
  //bool missingValues_;      ///< true if the method can handle missing values 
  // bool noClass_;            ///< true if the method can handle data without a given class (e.g. clustering)        
  //bool catClass_;           ///< true if the method can handle categorical classes 
  //bool binaryClassOnly_;    ///< true if the method can handle binary classes  
  // bool numClass_;           ///< true if the method can handle numeric classes  
  // bool missingClassValues_; ///< true if the method can handle missing values for the class
};


