/* Open source system for classification learning from very large data
** Class for an instance
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
#pragma once
#include <limits>
#include <vector>

class InstanceStream;

typedef unsigned int CategoricalAttribute;
typedef unsigned int NumericAttribute;
enum AttType {CATEGORICAL, NUMERIC, CLASS};

typedef unsigned int CatValue;
typedef float NumValue;

const CatValue ERROR = std::numeric_limits<CatValue>::max();

const NumValue MISSINGNUM = std::numeric_limits<NumValue>::max();

class instance
{
public:
  instance();
  instance(InstanceStream &is);
  ~instance(void);

  void init(InstanceStream &is);

  inline CatValue getCatVal(const CategoricalAttribute att) const { return catVals[att]; }
  inline NumValue getNumVal(const NumericAttribute att) const { return numVals[att]; }
  inline CatValue getClass() const { return theClass; }

  // true iff a numeric value is missing
  inline bool isMissing(const NumericAttribute a) const {
    return numVals[a] == MISSINGNUM;
  }

  friend class InstanceStream;  // allow InstanceStreams to access private members
  friend class InstanceStreamDiscretiser;

private:
  inline void setCatVal(const CategoricalAttribute att, const CatValue v) { catVals[att]=v; }
  inline void setNumVal(const NumericAttribute att, const NumValue v) { numVals[att]=v; }
  inline void setClass(const CatValue v) { theClass=v; }

  std::vector<CatValue> catVals;
  std::vector<NumValue> numVals;
  CatValue theClass;
};
