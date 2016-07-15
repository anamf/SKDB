/* Gigal: An open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
**
** Module for registering each of the available learners.
** WOuld ideally use a map from char* to learner constructors, but c++ does not seem to support that
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
#include "learnerRegistry.h"
#include "utils.h"

// learners
#include "kdb.h"
#include "kdbSelective.h"
#include "nb.h"
#include "tan.h"
#include "aode.h"

// create a new learner, selected by name
learner *createLearner(const char *learnername, char*const*& argv, char*const* end) {
  if (streq(learnername, "aode")) {
    return new aode(argv, end);
  }
  else if (streq(learnername, "kdb")) {
    return new kdb(argv, end);
  }
  else if (streq(learnername, "kdb-selective")) {
    return new kdbSelective(argv, end);
  }
  else if (streq(learnername, "nb")) {
    return new nb(argv, end);
  }
  else if (streq(learnername, "tan")) {
    return new TAN(argv, end);
  }
  return NULL;
}

