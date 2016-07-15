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

#include "capabilities.h"
#include "instanceStream.h"
#include "utils.h"

capabilities::capabilities() {
  catAtts_ = false;
  numAtts_ = false;
//  binaryClassOnly_ = false;
//  missingValues_ = false;
//  noClass_ = false;
//  catClass_ = false;
//  numClass_ = false;
//  missingClassValues_ = false;
}

capabilities::~capabilities() {
}


void capabilities::setCatAtts(bool catAtts){
  catAtts_ = catAtts;
}
  
void capabilities::setNumAtts(bool numAtts){
  numAtts_ = numAtts;
}

//void capabilities::setmissingValues(bool missingValues){
//  missingValues_ = missingValues;
//}
//
//void capabilities::setNoClass(bool noClass){
//  noClass_ = noClass;
//}
//
//void capabilities::setCatClass(bool catClass){
//  catClass_ = catClass;
//}
//
//void capabilities::setBinaryClassOnly(bool binaryClassOnly){
//  binaryClassOnly_ = binaryClassOnly;
//}
//
//void capabilities::setNumClass(bool numClass){
//  numClass_ = numClass;
//}
//
//void capabilities::setMissingClassValues_(bool missingClassValues){
//  missingClassValues_ = missingClassValues;
//}
//  
  
void capabilities::testCapabilities(InstanceStream &is, const std::string& methodName){
  
  if(!catAtts_ && is.getNoCatAtts())
    error("%s does not support categorical attributes",methodName.c_str());
  
  if(!numAtts_ && is.getNoNumAtts())
    error("%s does not support numeric attributes",methodName.c_str());
  
//  if(binaryClassOnly_ && is.getNoClasses()>2)
//    error("%s only supports binary-class problems",methodName.c_str());
}



