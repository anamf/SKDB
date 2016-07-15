/* Open source system for classification learning from very large data
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
#include "instanceFile.h"
#include "FilterSet.h"
#include "learner.h"


class TrainTestArgs {
public:
  TrainTestArgs() : calcAUPRC_(false) {}

  void getArgs(char*const*& argv, char*const* end);  // get settings from command line arguments

  bool calcAUPRC_;
};

/// train a learner from a training set and test against a test set read from a file
/// @param theLearner the learner to test
/// @param instStream the instance stream to train from
/// @param instFile the underlying instance file form which the stream is fed.  This file will be changed to the test file
/// @param filters the set of filters to apply before learning
/// @param testfilename the name of the file form which the test cases should be read. The source of instStream will be set to this file by this function.
void trainTest(learner *theLearner, InstanceStream &instStream, InstanceFile &instFile, FilterSet &filters, char * testfilename, const TrainTestArgs &args);
