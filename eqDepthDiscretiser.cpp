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
#include "eqDepthDiscretiser.h"
#include "utils.h"
#include <algorithm>
#include "math.h"
#include "globals.h"

eqDepthDiscretiser::eqDepthDiscretiser(char*const*& argv, char*const* end)
{
  intervals = 10;
  discType = specifiedIntervals;

  // get arguments
  while (argv != end) {
    if (*argv[0] == '-' && argv[0][1] == 'i') {
      getUIntFromStr(argv[0]+2, intervals, "i");
      ++argv;
    }
    else if (*argv[0] == '-' && argv[0][1] == 'r') {
      getUIntFromStr(argv[0]+2, root_, "r");
      discType = npkid;
      ++argv;
    }
    else if (streq(argv[0]+1, "-squared")) {
      discType = pkid;
      ++argv;
    }
    else if (streq(argv[0]+1, "-cubed")) {
      discType = pkidCubed;
      ++argv;
    }
    else {
      break;
    }
  }
}

eqDepthDiscretiser::~eqDepthDiscretiser(void)
{
}

void eqDepthDiscretiser::discretise(std::vector<NumValue> &vals, const std::vector<CatValue>& /*classes*/, unsigned int /*noOfClasses*/, std::vector<NumValue> &cuts) {
  if (vals.empty()) {
    cuts.push_back(0);
  }
  else {
    NumValue lastVal = std::numeric_limits<NumValue>::max();

    std::sort(vals.begin(), vals.end());

    // remove missing values from the array
    while (vals.back() == MISSINGNUM) vals.pop_back();

    if (intervals > vals.size()) 
      intervals = vals.size();

    if (intervals == 0) {
      // need to guard against all values being missing
      cuts.push_back(0);
    }
    else {
      double intervalSize;
      
      switch (discType) {
        case specifiedIntervals:
          intervalSize = vals.size()/static_cast<double>(intervals);
          break;
        case pkid:{
          intervals = static_cast<unsigned int>(floor(sqrt(static_cast<double>(vals.size()))));
          intervalSize = vals.size() / intervals;
          break;
        }
        case pkidCubed:{
          intervals = static_cast<unsigned int>(floor(pow(static_cast<double>(vals.size()), 1.0/3.0)));
          intervalSize = vals.size() / intervals;
          break;
        }
        case npkid:{
          intervals = static_cast<unsigned int>(floor(pow(static_cast<double>(vals.size()), 1.0/root_)));
          intervalSize = vals.size() / intervals;
          break;
        }
        default:
          error("discretisation type not supported");
      }
      if(verbosity>=2)
        printf("Interval size = %f\n",intervalSize);

      // find each of the cutpoints.
      // there is one less cutpoint than there are intervals
      for (unsigned int i = 0; i < intervals-1; i++) {
        const NumValue val = vals[static_cast<unsigned int>(intervalSize*(i+1))-1];
        if (val != lastVal && val < vals.back()) { // do not duplicate cuts and do not cut on the last value in the input
          cuts.push_back(val);
          lastVal = val;
        }
      }
    }
  }
}
