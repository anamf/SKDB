/* Open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
**
** Class for representing a class distribution.
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
#include "instanceStream.h"

class yDist
{
public:
  yDist();
  yDist(InstanceStream &is);
  ~yDist(void);

  void reset(const InstanceStream &is);

  void clear();
  void update(const instance &i);
  double p(CatValue y) const;
  double rawP(CatValue y) const;
  double ploocv(CatValue y, CatValue t) const; // used for leave-one-out-cv (t is removed)
  InstanceCount count(CatValue y) const;

  inline unsigned int getNoClasses() { return counts.size(); }

private:
  std::vector<InstanceCount> counts;
  InstanceCount total;

};

